#!/usr/bin/env python3
"""
predict_quality.py — JSON-based image quality predictor

Reads JSON metadata files from a digitized page archive and predicts whether
each associated image is Bad or OK/Good, using a Gradient Boosting regression
model trained on ground-truth labels.

The model predicts a continuous bad_probability (0-1) from 63 features
extracted entirely from the JSON file (no image pixels required). A tunable
decision threshold converts that probability into a binary Bad/OK label.

Usage:
    # Basic — process a directory of JSON files, write results to CSV
    python predict_quality.py /path/to/json/files -o results.csv

    # Parallel — use 32 worker processes
    python predict_quality.py /path/to/json/files -o results.csv -w 32

    # Custom threshold — flag more aggressively (higher recall, lower precision)
    python predict_quality.py /path/to/json/files -o results.csv -t 0.35

    # Verbose — print progress every 1000 files
    python predict_quality.py /path/to/json/files -o results.csv -v

    # Recursive — process JSON files in all subdirectories
    python predict_quality.py /path/to/root -o results.csv --recursive

Requirements:
    Python 3.8+
    scikit-learn >= 1.0
    numpy
    orjson (recommended, ~40% faster; falls back to stdlib json if missing)

    Install:  pip install scikit-learn numpy orjson

Model file:
    Expects quality_model_v2.pkl in the same directory as this script.
    The pickle contains the trained GradientBoostingRegressor, feature names,
    and the default threshold (0.46).

Performance (on held-out test set of 1,000 files):
    F1 = 0.832 | Recall = 80.1% | Precision = 86.5% | Accuracy = 83.9%

Scaling estimate (300M files):
    64 cores:  ~5.5 hours (orjson) / ~8.6 hours (stdlib json)
    32 cores: ~11.0 hours (orjson) / ~17.2 hours (stdlib json)
     8 cores: ~44.0 hours (orjson) / ~69.0 hours (stdlib json)
"""

import argparse
import csv
import json
import os
import pickle
import statistics
import sys
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

# ── Fast JSON parser (orjson is ~40% faster than stdlib json) ──
try:
    import orjson
    def _parse_json(data: bytes):
        return orjson.loads(data)
    _JSON_PARSER = "orjson"
except ImportError:
    def _parse_json(data: bytes):
        return json.loads(data)
    _JSON_PARSER = "json (install orjson for ~40% faster parsing)"


def _load_json(path):
    """Load JSON from a local file path."""
    with open(path, "rb") as f:
        return _parse_json(f.read())


def _load_json_s3(s3_path):
    """Stream JSON directly from S3 without saving to disk."""
    import threading
    # Thread-local boto3 client to avoid contention
    if not hasattr(_s3_local, "client"):
        import boto3
        from botocore.config import Config
        _s3_local.client = boto3.client(
            "s3",
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        )
    path = s3_path.removeprefix("s3://")
    bucket, _, key = path.partition("/")
    obj = _s3_local.client.get_object(Bucket=bucket, Key=key)
    return _parse_json(obj["Body"].read())


import threading
_s3_local = threading.local()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _safe_stdev(values):
    """Standard deviation that returns 0 for fewer than 2 values."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def extract_features(json_path):
    """
    Extract 63 predictive features from a single JSON metadata file.

    Features span five categories:
      - Page geometry (width, height, megapixels, aspect ratio)
      - OCR article-level stats (confidence mean/min/max/stdev/range)
      - Block-level layout (label distribution, block confidence)
      - Word-level OCR quality (word confidence stats, low/high %)
      - Corrections & NER (counts, ratios, entity types)

    Returns a dict keyed by feature name.
    """
    if str(json_path).startswith("s3://"):
        data = _load_json_s3(json_path)
    else:
        data = _load_json(json_path)

    features = {}

    # ── Page geometry ──
    w = data.get("width", 0)
    h = data.get("height", 0)
    features["width"] = w
    features["height"] = h
    features["megapixels"] = (w * h) / 1_000_000
    features["aspect_ratio"] = w / h if h > 0 else 0

    # ── Article-level OCR ──
    articles = data.get("ocr", [])
    features["article_count"] = len(articles)

    art_confidences = [a["confidence"] for a in articles if "confidence" in a]
    features["confidence_mean"] = (
        statistics.mean(art_confidences) if art_confidences else 0
    )
    features["confidence_stdev"] = _safe_stdev(art_confidences)
    features["confidence_min"] = min(art_confidences) if art_confidences else 0
    features["confidence_max"] = max(art_confidences) if art_confidences else 0
    features["confidence_range"] = (
        features["confidence_max"] - features["confidence_min"]
    )
    features["articles_low_conf_pct"] = (
        sum(1 for c in art_confidences if c < 0.5) / len(art_confidences) * 100
        if art_confidences
        else 0
    )

    # ── Language diversity ──
    langs = [a.get("language", "unknown") for a in articles]
    lang_counts = Counter(langs)
    features["language_count"] = len(lang_counts)
    features["english_pct"] = (
        lang_counts.get("en", 0) / len(langs) * 100 if langs else 0
    )

    # ── Block & word-level analysis ──
    all_block_labels = []
    all_block_confs = []
    all_word_confs = []
    all_line_sizes = []
    total_lines = 0
    total_words = 0
    total_blocks = 0
    blocks_per_article = []

    for art in articles:
        blocks = art.get("blocks", [])
        blocks_per_article.append(len(blocks))
        for block in blocks:
            total_blocks += 1
            all_block_labels.append(block.get("label", "unknown"))
            if "confidence" in block:
                all_block_confs.append(block["confidence"])
            for line in block.get("data", {}).get("lines", []):
                total_lines += 1
                if "size" in line:
                    all_line_sizes.append(line["size"])
                wc = line.get("word_confidences", [])
                all_word_confs.extend(wc)
                total_words += len(wc)

    features["total_blocks"] = total_blocks
    features["total_lines"] = total_lines
    features["total_words"] = total_words

    # Block label distribution
    label_counts = Counter(all_block_labels)
    for lbl in [
        "Body", "Title", "Advertising", "Picture",
        "Caption", "Artifact", "Byline", "Intermediate",
    ]:
        features[f"block_{lbl.lower()}_count"] = label_counts.get(lbl, 0)
        features[f"block_{lbl.lower()}_pct"] = (
            label_counts.get(lbl, 0) / total_blocks * 100
            if total_blocks > 0
            else 0
        )

    # Block confidence
    features["block_conf_mean"] = (
        statistics.mean(all_block_confs) if all_block_confs else 0
    )
    features["block_conf_stdev"] = _safe_stdev(all_block_confs)
    features["block_conf_min"] = min(all_block_confs) if all_block_confs else 0

    # Word confidence (critical signal)
    features["word_conf_mean"] = (
        statistics.mean(all_word_confs) if all_word_confs else 0
    )
    features["word_conf_stdev"] = _safe_stdev(all_word_confs)
    features["word_conf_min"] = min(all_word_confs) if all_word_confs else 0
    features["word_conf_median"] = (
        statistics.median(all_word_confs) if all_word_confs else 0
    )
    features["low_conf_words_pct"] = (
        sum(1 for w in all_word_confs if w < 50) / len(all_word_confs) * 100
        if all_word_confs
        else 0
    )
    features["very_low_conf_words_pct"] = (
        sum(1 for w in all_word_confs if w < 20) / len(all_word_confs) * 100
        if all_word_confs
        else 0
    )
    features["high_conf_words_pct"] = (
        sum(1 for w in all_word_confs if w >= 90) / len(all_word_confs) * 100
        if all_word_confs
        else 0
    )

    # Line size stats
    features["line_size_mean"] = (
        statistics.mean(all_line_sizes) if all_line_sizes else 0
    )
    features["line_size_stdev"] = _safe_stdev(all_line_sizes)
    features["line_size_max"] = max(all_line_sizes) if all_line_sizes else 0

    # Density ratios
    features["words_per_line"] = total_words / total_lines if total_lines > 0 else 0
    features["blocks_per_article"] = (
        statistics.mean(blocks_per_article) if blocks_per_article else 0
    )

    # ── Corrections ──
    corr_data = data.get("corr_data", [])
    features["correction_count"] = len(corr_data)
    features["corrections_per_article"] = (
        len(corr_data) / len(articles) if articles else 0
    )
    features["corrections_per_word"] = (
        len(corr_data) / total_words if total_words > 0 else 0
    )
    features["corrections_per_line"] = (
        len(corr_data) / total_lines if total_lines > 0 else 0
    )

    # ── Named Entity Recognition ──
    ner_data = data.get("ner_data", [])
    features["ner_count"] = len(ner_data)
    features["ner_per_article"] = len(ner_data) / len(articles) if articles else 0
    features["ner_per_word"] = len(ner_data) / total_words if total_words > 0 else 0

    ner_types = [
        item[1] for item in ner_data if isinstance(item, list) and len(item) > 1
    ]
    ner_type_counts = Counter(ner_types)
    for nt in ["Person", "Organization", "Location", "Date", "Money"]:
        features[f"ner_{nt.lower()}_count"] = ner_type_counts.get(nt, 0)

    # ── Derived ratios ──
    features["content_density"] = (
        total_words / features["megapixels"] if features["megapixels"] > 0 else 0
    )
    features["article_density"] = (
        len(articles) / features["megapixels"] if features["megapixels"] > 0 else 0
    )
    features["advertising_ratio"] = (
        label_counts.get("Advertising", 0) / total_blocks if total_blocks > 0 else 0
    )
    features["picture_ratio"] = (
        label_counts.get("Picture", 0) / total_blocks if total_blocks > 0 else 0
    )

    return features


# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION (for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level globals set by _init_worker so each subprocess has access
# to the model without re-pickling it on every task.
_model = None
_feature_names = None
_threshold = None


def _init_worker(model_path):
    """Initializer for each worker process — loads the model once."""
    global _model, _feature_names, _threshold
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    _model = data["model"]
    _feature_names = data["feature_names"]
    _threshold = data["threshold"]


def _process_file(json_path):
    """
    Process a single JSON file (local or s3://): extract features, predict, return result dict.
    Returns a dict with an 'error' key on failure.
    """
    try:
        feats = extract_features(json_path)
        x = np.array([[feats[fn] for fn in _feature_names]])
        prob = float(np.clip(_model.predict(x)[0], 0, 1))
        label = "Bad" if prob >= _threshold else "OK"

        return {
            "filename": json_path.split("/")[-1],
            "prediction": label,
            "bad_probability": round(prob, 4),
            "confidence_mean": round(feats["confidence_mean"], 4),
            "corrections_per_article": round(feats["corrections_per_article"], 2),
            "megapixels": round(feats["megapixels"], 2),
            "word_conf_mean": round(feats["word_conf_mean"], 1),
            "low_conf_words_pct": round(feats["low_conf_words_pct"], 1),
            "content_density": round(feats["content_density"], 1),
            "article_count": feats["article_count"],
        }
    except Exception as e:
        return {"filename": json_path.split("/")[-1], "error": str(e)}


def _process_batch(json_paths):
    """Process a batch of files — reduces per-call overhead for imap."""
    return [_process_file(p) for p in json_paths]


# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_json_files(input_dir, recursive=False):
    """Find all .json files in the input directory."""
    input_path = Path(input_dir)
    if recursive:
        return sorted([str(p) for p in input_path.rglob("*.json")])
    else:
        return sorted([str(p) for p in input_path.glob("*.json")])


def load_s3_paths(paths_file):
    """Load S3 paths from a file (like json_files.txt)."""
    with open(paths_file, encoding="utf-8") as f:
        return [
            l.strip() for l in f
            if l.strip() and not l.startswith("JsonPath") and l.strip().startswith("s3://")
        ]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict image quality (Bad vs OK) from JSON metadata files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_quality.py /data/json_files -o results.csv
  python predict_quality.py /data/json_files -o results.csv -w 32
  python predict_quality.py /data/json_files -o results.csv -t 0.35 -v
  python predict_quality.py /data/json_files -o results.csv --recursive
        """,
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="Directory containing local JSON files to process",
    )
    parser.add_argument(
        "--s3-paths",
        default=None,
        help="File containing S3 paths (e.g. json_files.txt). Streams JSONs directly from S3 — no download needed.",
    )
    parser.add_argument(
        "-o", "--output",
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help="Decision threshold for Bad classification (default: 0.46, from training). "
             "Lower = higher recall, more false positives. "
             "Higher = higher precision, more false negatives.",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU count)",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Path to model pickle (default: quality_model_v2.pkl in script directory)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress updates every 1,000 files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for JSON files recursively in subdirectories",
    )
    parser.add_argument(
        "--errors-file",
        default=None,
        help="Write filenames that failed processing to this file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Files per worker batch (default: 100). "
             "Larger batches reduce IPC overhead for very large runs.",
    )

    args = parser.parse_args()

    # ── Resolve model path ──
    if args.model:
        model_path = args.model
    else:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "quality_model_v2.pkl"
        )

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        print(
            "  Place quality_model_v2.pkl next to this script, or use -m /path/to/model.pkl",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Override threshold if requested ──
    if args.threshold is not None:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        model_data["threshold"] = args.threshold
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        with open(tmp.name, "wb") as f:
            pickle.dump(model_data, f)
        effective_model_path = tmp.name
        effective_threshold = args.threshold
    else:
        effective_model_path = model_path
        with open(model_path, "rb") as f:
            effective_threshold = pickle.load(f)["threshold"]

    # ── Discover JSON files (local dir or S3 paths file) ──
    if args.s3_paths:
        json_files = load_s3_paths(args.s3_paths)
        if not json_files:
            print(f"ERROR: No s3:// paths found in {args.s3_paths}", file=sys.stderr)
            sys.exit(1)
        input_label = f"{args.s3_paths} ({len(json_files):,} S3 paths)"
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"ERROR: Not a directory: {input_dir}", file=sys.stderr)
            sys.exit(1)
        json_files = discover_json_files(input_dir, recursive=args.recursive)
        if not json_files:
            print(f"ERROR: No .json files found in {input_dir}", file=sys.stderr)
            sys.exit(1)
        input_label = f"{input_dir} ({len(json_files):,} JSON files)"
    else:
        print("ERROR: Provide either input_dir or --s3-paths", file=sys.stderr)
        sys.exit(1)

    n_workers = args.workers or cpu_count()
    n_files = len(json_files)

    print(f"predict_quality.py")
    print(f"  Input:     {input_label}")
    print(f"  Output:    {args.output}")
    print(f"  Model:     {model_path}")
    print(f"  Threshold: {effective_threshold:.2f}")
    print(f"  Workers:   {n_workers}")
    print(f"  Parser:    {_JSON_PARSER}")
    if args.recursive:
        print(f"  Mode:      recursive")
    print()

    # ── Process files in parallel ──
    results = []
    errors = []
    start_time = time.time()

    # Chunk file list into batches for reduced IPC overhead
    batch_size = args.batch_size
    batches = [
        json_files[i : i + batch_size]
        for i in range(0, len(json_files), batch_size)
    ]

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(effective_model_path,),
    ) as pool:
        processed = 0
        for batch_results in pool.imap_unordered(_process_batch, batches):
            for result in batch_results:
                if "error" in result:
                    errors.append(result)
                else:
                    results.append(result)
                processed += 1

            if args.verbose and processed % 1000 < batch_size:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (n_files - processed) / rate if rate > 0 else 0
                bad_so_far = sum(1 for r in results if r["prediction"] == "Bad")
                print(
                    f"  [{processed:,}/{n_files:,}] "
                    f"{rate:,.0f} files/sec | "
                    f"ETA {eta / 60:.1f} min | "
                    f"Bad so far: {bad_so_far:,}"
                )

    elapsed = time.time() - start_time

    # ── Write output CSV ──
    fieldnames = [
        "filename", "prediction", "bad_probability",
        "confidence_mean", "corrections_per_article", "megapixels",
        "word_conf_mean", "low_conf_words_pct", "content_density",
        "article_count",
    ]

    results.sort(key=lambda r: r["bad_probability"], reverse=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Write errors file if requested ──
    if args.errors_file and errors:
        with open(args.errors_file, "w") as f:
            for err in errors:
                f.write(f"{err['filename']}\t{err['error']}\n")

    # ── Summary ──
    bad_count = sum(1 for r in results if r["prediction"] == "Bad")
    ok_count = len(results) - bad_count

    print(f"Complete in {elapsed:.1f}s ({n_files / elapsed:,.0f} files/sec)")
    print(f"  Processed: {len(results):,}")
    print(f"  Errors:    {len(errors):,}")
    pct_bad = (bad_count / len(results) * 100) if results else 0
    pct_ok  = (ok_count  / len(results) * 100) if results else 0
    print(f"  Bad:       {bad_count:,} ({pct_bad:.1f}%)")
    print(f"  OK:        {ok_count:,} ({pct_ok:.1f}%)")
    print(f"  Output:    {args.output}")

    # Cleanup temp model if created
    if args.threshold is not None:
        os.unlink(effective_model_path)


if __name__ == "__main__":
    main()
