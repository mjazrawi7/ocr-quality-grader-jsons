# json-quality-grader

A pipeline for bulk-downloading and quality-grading OCR JSON files at scale using a pre-trained machine learning model. Predicts whether each digitized page is `Bad` or `OK` based on 63 features extracted from the JSON metadata — no image pixels required.

Supports two modes:
- **Local mode**: process downloaded JSON files from a folder
- **S3 streaming mode**: stream directly from S3 — no download required (recommended for large-scale runs)

## Project Structure

```
├── predict_quality.py        # Main processing script
├── download_jsons.py         # Bulk S3 downloader (parallel, resumable)
├── quality_model_v2.pkl      # Pre-trained GradientBoosting model
├── json_files.txt            # List of S3 paths to process
├── requirements.txt          # Python dependencies
└── output/                   # Results and summaries written here
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\Activate.ps1     # Windows

pip install -r requirements.txt
```

## Usage

### Option A — Stream directly from S3 (recommended for EC2 / large scale)

No download needed. Reads each JSON directly from S3 using per-process boto3 clients:

```bash
python predict_quality.py --s3-paths json_files.txt -w 128 --output output/results_batch1.csv -v
```

`json_files.txt` should be a plain text file with one `s3://bucket/path/file.json` per line (a header line `JsonPath` is automatically skipped if present).

### Option B — Process local files

```bash
python predict_quality.py <input_folder> --output output/results.csv -w 8 -v
```

### All arguments

| Argument | Description | Default |
|---|---|---|
| `input_folder` | Folder containing local JSON files | optional |
| `--s3-paths` | Text file of S3 paths (one per line) | optional |
| `--output` / `-o` | Output CSV path | `output/results.csv` |
| `-w` | Number of parallel workers | CPU count |
| `-v` | Verbose progress output | off |
| `-t` | Decision threshold (0–1) | 0.46 |
| `--recursive` | Search subdirectories (local mode) | off |
| `--batch-size` | Files per work chunk | 100 |

### Download JSON files from S3 (Option B only)

```bash
python download_jsons.py
```
Reads S3 paths from `json_files.txt`, downloads in parallel (100 threads). Resumable — skips already-downloaded files. Retry failures:
```bash
python download_jsons.py --input download_errors.txt
```

## Output

Each run produces two files based on `--output`:

- **`results_batchN.csv`** — one row per file, sorted by `bad_probability` descending
- **`results_batchN_summary.txt`** — counts of Bad/OK/Errors for that batch

| Column | Description |
|---|---|
| `filename` | JSON file name or S3 path |
| `prediction` | `OK` or `Bad` |
| `bad_probability` | Model confidence (0–1) |
| `confidence_mean` | Mean OCR article confidence |
| `word_conf_mean` | Mean word-level confidence |
| `low_conf_words_pct` | % of low confidence words |
| `corrections_per_article` | Avg corrections per article |
| `megapixels` | Page image size |
| `content_density` | Content density score |
| `article_count` | Number of articles on page |

### Merging batch outputs

```bash
head -1 output/results_batch1.csv > output/results_all.csv
tail -n +2 -q output/results_batch*.csv >> output/results_all.csv
```

## Model Performance

Trained on ground-truth labels, evaluated on a held-out test set of 1,000 files:

| Metric | Score |
|---|---|
| F1 | 0.832 |
| Recall | 80.1% |
| Precision | 86.5% |
| Accuracy | 83.9% |

Observed Bad rate on production data: **~34.2%**

## Scaling to 20M Records (EC2)

Recommended: **`c7i.8xlarge`** (32 vCPUs, 64GB RAM, ~$1.36/hr)

### Measured throughput on c7i.8xlarge

| Workers | files/sec | 20M estimate |
|---------|-----------|--------------|
| 32 | 256 | ~21.7 hrs |
| 64 | 672 | ~8.3 hrs |
| 128 | **948** | **~5.9 hrs** |

### Recommended batch workflow

Split the full path list into batches of ~4-5M to limit re-run cost on failure. Run inside `tmux` so SSH disconnection doesn't interrupt the job:

```bash
sudo dnf install -y tmux          # Amazon Linux 2023
tmux new -s batches

python predict_quality.py --s3-paths batch1.txt -w 128 --output output/results_batch1.csv -v && \
python predict_quality.py --s3-paths batch2.txt -w 128 --output output/results_batch2.csv -v && \
python predict_quality.py --s3-paths batch3.txt -w 128 --output output/results_batch3.csv -v && \
python predict_quality.py --s3-paths batch4.txt -w 128 --output output/results_batch4.csv -v

# Ctrl+B, D  →  detach and go to bed
```

Reattach later:
```bash
tmux attach -t batches
```

## Requirements

- Python 3.8+
- AWS credentials configured (`aws configure`)
- Read access to the S3 buckets listed in `json_files.txt`
