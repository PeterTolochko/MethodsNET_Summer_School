# Advanced Text Analysis Summer School: Code Materials

This folder contains the student-facing code materials for the five-day summer school.

## Structure

```text
install_dependencies.py
materials/
  data/
    README.md
    federalist.csv
    headlines.csv
    reviews.csv
    sms_spam.csv
    sotu.csv
    tweet_eval_sentiment_sample.csv
    twenty_newsgroups_sample.csv
  notebooks/
    day_1_representation.ipynb
    day_2_labels_and_classification.ipynb
    day_3_unsupervised_and_topics.ipynb
    day_4_embeddings_and_llms.ipynb
    day_4b_transformer_classification_and_finetuning.ipynb
    day_5_research_design_lab.ipynb
    day_5b_causal_inference_with_text.ipynb
  requirements.txt
  requirements-transformers.txt
  environment.yml
  environment-transformers.yml
  scripts/
    download_open_datasets.py
```

## Setup

### Prerequisites

- Python 3.10 or newer.
- Internet access during installation, because the installer downloads Python packages and the spaCy English model.
- A terminal or command prompt opened at the repository root.
- Enough disk space for the optional transformer stack. The default install includes torch and Hugging Face packages; use `--base-only` if students will not run the Day 4B transformer model sections.

### Recommended install

The easiest setup is to run the single installer from the repository root:

```bash
python3 install_dependencies.py
```

This creates `.venv`, installs the base notebook stack, installs the optional transformer stack for the BERT/RoBERTa notebook, downloads the spaCy `en_core_web_sm` model, and registers a Jupyter kernel called `Python (MethodsNET 2026)`.

On Windows, use:

```bash
python install_dependencies.py
```

Then start Jupyter:

```bash
source .venv/bin/activate
jupyter lab materials/notebooks
```

On Windows, activate with:

```bash
.venv\Scripts\activate
jupyter lab materials/notebooks
```

For a lighter install without the transformer packages:

```bash
python3 install_dependencies.py --base-only
```

If you already have a Python environment active and do not want a local `.venv`:

```bash
python3 install_dependencies.py --no-venv
```

The notebooks use a small Python stack: pandas, numpy, scikit-learn, spaCy, matplotlib, seaborn, and Jupyter. Day 4B additionally uses torch, transformers, accelerate, and datasets.

### Running the notebooks

1. Open JupyterLab with `jupyter lab materials/notebooks`.
2. Open the notebook for the relevant day.
3. If Jupyter asks for a kernel, choose `Python (MethodsNET 2026)`. If you installed with `--base-only`, choose `Python (MethodsNET 2026 base)`.
4. Run cells from top to bottom. The notebooks assume earlier setup cells have already been run.
5. The datasets are already included in `materials/data/`; students do not need to download separate data files.

### Local datasets

The teaching notebooks use local CSV files only. The richer downloaded datasets are:

- `sms_spam.csv`: SMS Spam Collection for binary classification, imbalance, threshold choice, and transformer fine-tuning.
- `twenty_newsgroups_sample.csv`: sampled 20 Newsgroups posts for topic modeling, clustering, and validation against known source categories.
- `tweet_eval_sentiment_sample.csv`: sampled TweetEval sentiment posts for short-text sentiment, embeddings, and pretrained transformer demonstrations.

Dataset provenance is documented in `materials/data/README.md`. To rebuild the downloaded CSVs, run:

```bash
python3 materials/scripts/download_open_datasets.py --force
```

Rebuilding TweetEval requires the optional transformer environment because it uses the Hugging Face `datasets` package. To rebuild only the non-Hugging Face datasets, use:

```bash
python3 materials/scripts/download_open_datasets.py --force --skip-tweet-eval
```

### Transformer notebook

`day_4b_transformer_classification_and_finetuning.ipynb` is optional and heavier than the other notebooks. It contains real Hugging Face code for BERT/RoBERTa-style models, but download-heavy cells are disabled by default.

To run those sections, install the full environment with:

```bash
python3 install_dependencies.py
```

Then open Day 4B and change the relevant flags near the top of the notebook:

```python
RUN_TRANSFORMER_TOKENIZATION = True
RUN_PRETRAINED_PIPELINE = True
RUN_FROZEN_FEATURES = True
RUN_FINE_TUNING = True
```

For a shorter classroom demonstration, turn on one flag at a time. Fine-tuning can be slow on a laptop CPU. To show BERT or RoBERTa specifically, change:

```python
MODEL_NAME = 'bert-base-uncased'
# or
MODEL_NAME = 'roberta-base'
```

### Troubleshooting

- `python3: command not found`: try `python install_dependencies.py`, or install Python 3.10+ first.
- `No module named ...`: make sure the `.venv` is activated and that Jupyter is using the `Python (MethodsNET 2026)` kernel.
- spaCy model download fails: rerun `python -m spacy download en_core_web_sm` inside the activated environment. The notebooks still run with a blank tokenizer if the model is missing.
- Hugging Face model download fails: check internet access and rerun only the Day 4B cell that failed.
- Installation is too slow or too large: use `python3 install_dependencies.py --base-only` and skip Day 4B model execution.

### Manual setup

Manual setup remains possible.

With pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r materials/requirements-transformers.txt
python -m spacy download en_core_web_sm
jupyter lab materials/notebooks
```

For base notebooks only:

```bash
pip install -r materials/requirements.txt
```

With conda or mamba:

```bash
conda env create -f materials/environment.yml
conda activate methodsnet-text-analysis
jupyter lab materials/notebooks
```

Optional, but recommended for richer spaCy examples:

```bash
python -m spacy download en_core_web_sm
```

For the optional BERT/RoBERTa fine-tuning notebook:

```bash
conda env create -f materials/environment-transformers.yml
conda activate methodsnet-text-analysis-transformers
jupyter lab materials/notebooks
```
