# GROUP_N — Mindful Consumption NLP Pipeline

Inverted-purpose adaptation of `koosha-t/Sentiment-Analysis-NLP-for-Marketting`. Detects and classifies desire-language, FOMO, advertising echo, and impulsive-purchase signals in Amazon product reviews.

## Layout

| Path | Purpose |
|---|---|
| `requirements.txt` | All Python dependencies. Install with `pip install -r requirements.txt`. |
| `utils.py` | Shared paths, seed control, dataset locator. |
| `data/raw/` | Place the Kaggle `kritanjalijain/amazon-reviews` `train.csv` / `test.csv` here. |
| `data/artifacts/` | Auto-populated parquet/numpy artifacts produced by the notebooks. |
| `figures/` | Auto-populated figures for the LaTeX report. |
| `00_setup_and_data_audit.ipynb` | Verify dataset shape, schema, polarity distribution, length distribution. **Run first.** |
| `01_preprocessing_and_sampling.ipynb` | Clean text; build stratified 100K / 200K / 500K samples; train/val/test split. |
| `02_feature_extraction.ipynb` | TF-IDF (sparse) + sentence-transformer embeddings (dense). |
| `03_lexicon_tier.ipynb` | VADER + AFINN + (optional) NRC-EmoLex / NRC-VAD desire signature. |
| `04_classical_ml_tier.ipynb` | LogisticRegression → LinearSVC → LightGBM on TF-IDF, with imbalanced-learn comparison. |
| `05_neural_tier_finetune.ipynb` | DistilBERT fine-tune via HuggingFace `Trainer` (GPU recommended). |
| `06_zero_shot_desire_categories.ipynb` | Zero-shot labeling: status-seeking, FOMO, genuine-need, hedonic-impulse, etc. |
| `07_topic_modeling.ipynb` | BERTopic + KeyBERT, with Gensim LDA as comparison baseline. |
| `08_evaluation_and_figures.ipynb` | Compile cross-tier results, generate publication-quality figures. |
| `build_notebooks.py` | Regenerates the notebooks from source. Optional. |

## Dataset

Download the Kaggle dataset at <https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews> and unzip into `data/raw/` so that `data/raw/train.csv` (and `test.csv`) exist. Expected schema: `polarity` (1=negative, 2=positive), `title`, `text`. **Notebook `00` reconciles the actual size and schema with the project brief.**

## Run order

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt')"

jupyter lab  # then run notebooks 00 → 08 in order
```

Each notebook reads the previous stage's artifact from `data/artifacts/` and writes its own. Re-running a notebook idempotently overwrites its outputs.

## Compute notes

| Notebook | Typical runtime | Hardware |
|---|---|---|
| 00 – 04 | seconds – minutes | CPU only |
| 05 (DistilBERT fine-tune) | 30–60 min on free Colab T4 | GPU recommended |
| 06 (zero-shot DeBERTa) | 5–30 min on a small sample | GPU recommended |
| 07 (BERTopic) | 1–5 min on 100K rows | CPU OK if embeddings already cached |
| 08 | instant | CPU |

Sample-size constants live at the top of each notebook — shrink for faster iteration.

## License notes for downstream submission

NRC EmoLex / NRC VAD lexicons are free for academic research but require a separate commercial license; cite them rather than redistributing the lexicon files.
