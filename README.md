# json-quality-grader

A pipeline for bulk-downloading and quality-grading OCR JSON files at scale using a pre-trained machine learning model. Predicts whether each digitized page is `Bad` or `OK` based on 63 features extracted from the JSON metadata — no image pixels required.

## Project Structure

```
├── predict_quality.py        # Main processing script
├── download_jsons.py         # Bulk S3 downloader (parallel, resumable)
├── quality_model_v2.pkl      # Pre-trained GradientBoosting model
├── json_files.txt            # List of S3 paths to process
└── requirements.txt          # Python dependencies
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\Activate.ps1     # Windows

pip install -r requirements.txt
```

## Usage

### Download JSON files from S3
```bash
python download_jsons.py
```
Reads S3 paths from `json_files.txt` and downloads in parallel (100 threads by default). Resumable — skips already-downloaded files. Failed downloads saved to `download_errors.txt` for retry:
```bash
python download_jsons.py --input download_errors.txt
```

### Run quality prediction
```bash
python predict_quality.py <input_folder> -o results.csv -w 8 -v
```

| Argument | Description | Default |
|---|---|---|
| `input_folder` | Folder containing JSON files | required |
| `-o` | Output CSV file | required |
| `-w` | Number of parallel workers | all CPUs |
| `-v` | Verbose progress output | off |
| `-t` | Decision threshold (0–1) | 0.46 |
| `--recursive` | Search subdirectories | off |

## Output

`results.csv` — one row per file, sorted by `bad_probability` descending:

| Column | Description |
|---|---|
| `filename` | JSON file name |
| `prediction` | `OK` or `Bad` |
| `bad_probability` | Model confidence (0–1) |
| `confidence_mean` | Mean OCR article confidence |
| `word_conf_mean` | Mean word-level confidence |
| `low_conf_words_pct` | % of low confidence words |
| `corrections_per_article` | Avg corrections per article |
| `megapixels` | Page image size |
| `content_density` | Content density score |
| `article_count` | Number of articles on page |

## Model Performance

Trained on ground-truth labels, evaluated on a held-out test set of 1,000 files:

| Metric | Score |
|---|---|
| F1 | 0.832 |
| Recall | 80.1% |
| Precision | 86.5% |
| Accuracy | 83.9% |

## Scaling to 20M Records (EC2)

Recommended: **`c7i.8xlarge`** (32 vCPUs, 64GB RAM, ~$1.36/hr)

- Point `json_files.txt` at your full S3 path list
- No need to download files to EC2 — reads directly from S3
- Write output CSV to local EBS (`gp3`), push to S3 when done
- Run inside `tmux` to avoid disconnection issues
- Expected throughput: ~1,500–2,000 files/sec → ~3 hrs for 20M files

## Requirements

- Python 3.8+
- AWS credentials configured (`aws configure`)
- Read access to the S3 buckets listed in `json_files.txt`
