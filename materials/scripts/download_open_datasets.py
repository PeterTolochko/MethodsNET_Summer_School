#!/usr/bin/env python3
"""Download and prepare local classroom datasets for the text analysis notebooks.

The notebooks should not depend on live network access. This script downloads a
few open, reusable text corpora once and saves compact CSV files in
materials/data/.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "materials" / "data"
RAW_DIR = ROOT / "materials" / "raw_downloads"
RANDOM_STATE = 42

SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

NEWSGROUP_CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
]

TWEET_EVAL_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def save_csv(df: pd.DataFrame, path: Path, force: bool) -> None:
    if path.exists() and not force:
        print(f"exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote: {path} ({len(df):,} rows)")


def sample_per_group(df: pd.DataFrame, group_col: str, n: int) -> pd.DataFrame:
    pieces = []
    for _, group in df.groupby(group_col):
        pieces.append(group.sample(min(len(group), n), random_state=RANDOM_STATE))
    return pd.concat(pieces, ignore_index=True)


def download_sms_spam(force: bool) -> None:
    out_path = DATA_DIR / "sms_spam.csv"
    if out_path.exists() and not force:
        print(f"exists: {out_path}")
        return

    print("downloading SMS Spam Collection...")
    with urlopen(SMS_URL, timeout=60) as response:
        payload = response.read()

    raw_zip = RAW_DIR / "sms_spam_collection.zip"
    raw_zip.parent.mkdir(parents=True, exist_ok=True)
    raw_zip.write_bytes(payload)

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with zf.open("SMSSpamCollection") as handle:
            sms = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=["label", "text"],
                quoting=csv.QUOTE_NONE,
                encoding="utf-8",
            )

    sms["text"] = sms["text"].map(normalize_space)
    sms["label"] = sms["label"].str.lower()
    sms["is_spam"] = (sms["label"] == "spam").astype(int)
    sms["source"] = "UCI SMS Spam Collection"
    sms["source_url"] = SMS_URL
    sms = sms[["text", "label", "is_spam", "source", "source_url"]]
    save_csv(sms, out_path, force=True)


def download_twenty_newsgroups(force: bool, per_category: int) -> None:
    out_path = DATA_DIR / "twenty_newsgroups_sample.csv"
    if out_path.exists() and not force:
        print(f"exists: {out_path}")
        return

    print("downloading 20 Newsgroups via scikit-learn...")
    from sklearn.datasets import fetch_20newsgroups

    data_home = RAW_DIR / "scikit_learn"
    bunch = fetch_20newsgroups(
        subset="all",
        categories=NEWSGROUP_CATEGORIES,
        remove=("headers", "footers", "quotes"),
        data_home=data_home,
    )

    rows = pd.DataFrame(
        {
            "text": [normalize_space(text) for text in bunch.data],
            "target": bunch.target,
        }
    )
    rows["category"] = rows["target"].map(lambda i: bunch.target_names[int(i)])
    rows = rows[rows["text"].str.len().between(120, 6000)].copy()

    sampled = (
        sample_per_group(rows, "category", per_category)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    sampled["source"] = "20 Newsgroups"
    sampled["source_url"] = "https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset"
    sampled = sampled[["text", "category", "target", "source", "source_url"]]
    save_csv(sampled, out_path, force=True)


def download_tweet_eval(force: bool, per_label_per_split: int) -> None:
    out_path = DATA_DIR / "tweet_eval_sentiment_sample.csv"
    if out_path.exists() and not force:
        print(f"exists: {out_path}")
        return

    print("downloading TweetEval sentiment via Hugging Face datasets...")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required for TweetEval. "
            "Install materials/requirements-transformers.txt and rerun."
        ) from exc

    hf_cache = RAW_DIR / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_cache))
    dataset = load_dataset(
        "cardiffnlp/tweet_eval",
        "sentiment",
        cache_dir=str(hf_cache),
    )

    pieces = []
    for split_name, split in dataset.items():
        df = split.to_pandas()
        df["split"] = split_name
        df["label_name"] = df["label"].map(TWEET_EVAL_LABELS)
        df["text"] = df["text"].map(normalize_space)
        df = df[df["text"].str.len().between(10, 500)].copy()
        sampled = sample_per_group(df, "label_name", per_label_per_split)
        pieces.append(sampled)

    tweets = pd.concat(pieces, ignore_index=True)
    tweets = tweets.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    tweets["source"] = "TweetEval sentiment"
    tweets["source_url"] = "https://huggingface.co/datasets/cardiffnlp/tweet_eval"
    tweets = tweets[["text", "label", "label_name", "split", "source", "source_url"]]
    save_csv(tweets, out_path, force=True)


def write_dataset_readme() -> None:
    path = DATA_DIR / "README.md"
    content = """# Local Text Datasets

These files are bundled so students can run the notebooks without downloading
data during class.

| File | Main use | Source |
| --- | --- | --- |
| `sms_spam.csv` | Binary classification, imbalance, threshold choice, transformer fine-tuning | UCI SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| `twenty_newsgroups_sample.csv` | Topic modeling, clustering, embeddings, multiclass classification | 20 Newsgroups via scikit-learn: https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset |
| `tweet_eval_sentiment_sample.csv` | Short-text sentiment examples and optional transformer experiments | TweetEval sentiment: https://huggingface.co/datasets/cardiffnlp/tweet_eval |
| `federalist.csv` | Authorship attribution and stylometry | Legacy course material |
| `headlines.csv` | Dictionary measurement and source comparison | Legacy course material |
| `reviews.csv` | Sentiment classification fallback | Legacy course material |
| `sotu.csv` | Political speech representation and temporal analysis | Legacy course material |

The downloaded files are classroom-sized subsets where needed. The raw download
caches live under `materials/raw_downloads/` and can be deleted after rebuilding
the CSVs.
"""
    path.write_text(content, encoding="utf-8")
    print(f"wrote: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing local CSVs.",
    )
    parser.add_argument(
        "--skip-tweet-eval",
        action="store_true",
        help="Skip the Hugging Face TweetEval download.",
    )
    parser.add_argument(
        "--newsgroups-per-category",
        type=int,
        default=200,
        help="Rows to sample from each selected 20 Newsgroups category.",
    )
    parser.add_argument(
        "--tweets-per-label-per-split",
        type=int,
        default=300,
        help="Rows to sample from each TweetEval sentiment label within each split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    download_sms_spam(force=args.force)
    download_twenty_newsgroups(
        force=args.force,
        per_category=args.newsgroups_per_category,
    )
    if not args.skip_tweet_eval:
        download_tweet_eval(
            force=args.force,
            per_label_per_split=args.tweets_per_label_per_split,
        )
    write_dataset_readme()


if __name__ == "__main__":
    main()
