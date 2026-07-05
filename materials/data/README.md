# Local Text Datasets

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
