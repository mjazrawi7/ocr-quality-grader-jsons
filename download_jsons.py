"""
Fastest parallel S3 download script.
Uses ThreadPoolExecutor with high concurrency to saturate network bandwidth.
"""

import boto3
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from botocore.config import Config

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_FILE  = "json_files.txt"
OUTPUT_DIR  = "downloaded_jsons"
MAX_WORKERS = 100        # Threads; increase if network allows (try 150-200)
FLAT_OUTPUT = True       # True  → all files in OUTPUT_DIR (fast, no subdirs)
                         # False → mirror the S3 key structure under OUTPUT_DIR
# ─────────────────────────────────────────────────────────────────────────────

# Boto3 client pool (one per thread to avoid contention)
_local = threading.local()

def get_client():
    if not hasattr(_local, "s3"):
        _local.s3 = boto3.client(
            "s3",
            config=Config(
                max_pool_connections=MAX_WORKERS,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
    return _local.s3


def parse_s3_path(s3_path: str):
    """s3://bucket/key/... → (bucket, key)"""
    path = s3_path.removeprefix("s3://")
    bucket, _, key = path.partition("/")
    return bucket, key


def download_one(s3_path: str, output_dir: Path, flat: bool):
    bucket, key = parse_s3_path(s3_path.strip())
    if flat:
        dest = output_dir / Path(key).name
    else:
        dest = output_dir / key

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    if dest.exists():
        return s3_path, "skipped"

    get_client().download_file(bucket, key, str(dest))
    return s3_path, "ok"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="Path to input file (overrides INPUT_FILE). "
                             "Supports plain S3-path lists OR tab-separated "
                             "error files (first column used).")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    input_path = Path(args.input) if args.input else base_dir / INPUT_FILE
    output_dir = base_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        lines = [
            l.split("\t")[0].strip()          # handle tab-separated error files
            for l in f
            if l.strip() and not l.startswith("JsonPath")
        ]

    total = len(lines)
    print(f"Files to download : {total:,}")
    print(f"Workers           : {MAX_WORKERS}")
    print(f"Output directory  : {output_dir}\n")

    done = 0
    errors = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, p, output_dir, FLAT_OUTPUT): p for p in lines}

        for future in as_completed(futures):
            try:
                path, status = future.result()
            except Exception as exc:
                path = futures[future]
                errors.append((path, str(exc)))
                status = "error"

            with lock:
                done += 1
                if done % 500 == 0 or done == total:
                    pct = done / total * 100
                    print(f"  [{done:>6}/{total}]  {pct:5.1f}%   errors so far: {len(errors)}")

    print(f"\nDone. {total - len(errors):,} downloaded, {len(errors):,} errors.")

    if errors:
        err_file = base_dir / "download_errors.txt"
        with open(err_file, "w", encoding="utf-8") as f:
            for path, msg in errors:
                f.write(f"{path}\t{msg}\n")
        print(f"Error list saved to: {err_file}")


if __name__ == "__main__":
    main()
