"""Generates the nine Jupyter notebooks for the Mindful-Consumption NLP pipeline.

Run: python build_notebooks.py
Idempotent: overwrites any existing .ipynb files.
"""
from __future__ import annotations

import json
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent


def md(text: str):
    return ("markdown", text)


def code(text: str):
    return ("code", text)


def make_notebook(cells):
    nb_cells = []
    for ctype, src in cells:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]] if len(lines) > 1 else [src]
        if source == [""]:
            source = []
        cell = {"cell_type": ctype, "metadata": {}, "source": source}
        if ctype == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(name: str, cells):
    path = CODE_DIR / name
    nb = make_notebook(cells)
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path.relative_to(CODE_DIR)}")


# ---------------------------------------------------------------------------
# 00 — Setup & Data Audit
# ---------------------------------------------------------------------------

NB00 = [
    md("""# 00 — Setup & Data Audit

Verifies the Kaggle Amazon Reviews dataset is present, prints schema/shape/distribution stats, and saves the first descriptive figures. **Run before anything else.**

Expected file: `data/raw/train.csv` from <https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews>. Schema: `polarity` (1=neg, 2=pos), `title`, `text`."""),
    code("""import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from utils import RAW_DIR, FIGURES_DIR, find_amazon_csv, set_seed

sns.set_theme(style="whitegrid")
set_seed()"""),
    code("""csv_path = find_amazon_csv()
if csv_path is None:
    raise FileNotFoundError(
        f"Could not find train.csv under {RAW_DIR}. "
        "Download the Kaggle dataset and unzip into data/raw/."
    )
print(f"Loading {csv_path}")"""),
    code("""# Polars handles the file fast and lazily; the Kaggle CSV has no header row.
df = pl.read_csv(csv_path, has_header=False, new_columns=["polarity", "title", "text"])
print("Shape:", df.shape)
print("Schema:", df.schema)
df.head(3)"""),
    code("""print("Polarity counts:")
print(df["polarity"].value_counts().sort("polarity"))
print()
print("Null counts:")
print(df.null_count())"""),
    code("""# Length distributions
df = df.with_columns([
    df["title"].fill_null("").str.len_chars().alias("title_len"),
    df["text"].fill_null("").str.len_chars().alias("text_len"),
    df["text"].fill_null("").str.split(" ").list.len().alias("text_words"),
])
print(df.select(["title_len", "text_len", "text_words"]).describe())"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))

pol_counts = df["polarity"].value_counts().sort("polarity").to_pandas()
sns.barplot(data=pol_counts, x="polarity", y="count", ax=axes[0], color="#3b7dd8")
axes[0].set_title("Polarity distribution")
axes[0].set_xlabel("polarity (1 = negative, 2 = positive)")

# Cap at 1500 chars for a readable histogram
text_len_pd = df["text_len"].to_pandas().clip(upper=1500)
sns.histplot(text_len_pd, bins=60, ax=axes[1], color="#d87a3b")
axes[1].set_title("Review length (chars, capped at 1500)")
axes[1].set_xlabel("characters")

plt.tight_layout()
fig_path = FIGURES_DIR / "00_data_audit.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved {fig_path}")"""),
    md("""## Findings to record in the LaTeX Dataset section

- Actual row count: _fill from `df.shape` above_
- Polarity balance: _fill from value_counts above_
- Median review length (chars / words): _fill from describe()_
- Reconcile against project brief (which states ~34.7M rows). The canonical kritanjalijain Kaggle entry is the **Zhang et al. (2015) 4M binary polarity subset** — if the count above is ~4M, this is expected; if ~34M, you have a different mirror.
"""),
]


# ---------------------------------------------------------------------------
# 01 — Preprocessing & Sampling
# ---------------------------------------------------------------------------

NB01 = [
    md("""# 01 — Preprocessing & Sampling

Builds three stratified samples (debug / classical / neural) from the raw corpus, applies light text cleaning, and writes parquet artifacts that downstream notebooks consume.

Sample sizes are configurable below. Defaults are conservative for a free-tier laptop / Colab session."""),
    code("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import re
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import ARTIFACTS_DIR, RANDOM_STATE, find_amazon_csv, set_seed

set_seed()

# Sample sizes — shrink for faster iteration
N_DEBUG    = 100_000   # quick lexicon + classical demo
N_CLASSICAL = 500_000  # full classical-ML reporting
N_NEURAL   = 200_000   # transformer fine-tuning"""),
    code("""csv_path = find_amazon_csv()
df = pl.read_csv(csv_path, has_header=False, new_columns=["polarity", "title", "text"])
df = df.drop_nulls(subset=["polarity", "text"])
print("Loaded:", df.shape)"""),
    code("""URL_RE = re.compile(r"https?://\\S+|www\\.\\S+")
WS_RE = re.compile(r"\\s+")

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = s.replace("&quot;", '"').replace("&amp;", "&")
    s = WS_RE.sub(" ", s).strip()
    return s

# Pull title + text into a single 'review' field (matches reference repo style)
df = df.with_columns([
    (pl.col("title").fill_null("") + ". " + pl.col("text").fill_null("")).alias("raw")
])
df = df.with_columns(
    pl.col("raw").map_elements(clean_text, return_dtype=pl.Utf8).alias("review")
)
df = df.with_columns((pl.col("polarity") - 1).alias("label"))  # {0, 1}
print(df.select(["polarity", "label", "review"]).head(3))"""),
    code("""def stratified_sample(df_pl: pl.DataFrame, n: int, seed: int = RANDOM_STATE) -> pl.DataFrame:
    pdf = df_pl.to_pandas()
    sampled, _ = train_test_split(
        pdf, train_size=n, stratify=pdf["label"], random_state=seed
    )
    return pl.from_pandas(sampled.reset_index(drop=True))

# Build the three samples; write each as parquet
samples = {
    "debug":     stratified_sample(df, N_DEBUG),
    "classical": stratified_sample(df, N_CLASSICAL),
    "neural":    stratified_sample(df, N_NEURAL),
}
for name, sample in samples.items():
    out = ARTIFACTS_DIR / f"sample_{name}.parquet"
    sample.write_parquet(out)
    print(f"  {name:9s} -> {out.name}  ({sample.shape[0]:>8,} rows)")"""),
    code("""# Train / val / test split on the *neural* sample (used by every later notebook).
neural = samples["neural"].to_pandas()
train, temp = train_test_split(
    neural, test_size=0.20, stratify=neural["label"], random_state=RANDOM_STATE
)
val, test = train_test_split(
    temp, test_size=0.50, stratify=temp["label"], random_state=RANDOM_STATE
)
for name, split in [("train", train), ("val", val), ("test", test)]:
    out = ARTIFACTS_DIR / f"split_{name}.parquet"
    pl.from_pandas(split.reset_index(drop=True)).write_parquet(out)
    print(f"  {name:5s} -> {out.name}  ({len(split):>7,} rows)")"""),
    md("""## Artifacts produced

- `sample_debug.parquet` — 100K stratified
- `sample_classical.parquet` — 500K stratified
- `sample_neural.parquet` — 200K stratified
- `split_train.parquet`, `split_val.parquet`, `split_test.parquet` — 80/10/10 split of `sample_neural`

Downstream notebooks default to `split_*.parquet`. Swap to the larger `sample_classical.parquet` for the final classical-ML reporting run."""),
]


# ---------------------------------------------------------------------------
# 02 — Feature Extraction
# ---------------------------------------------------------------------------

NB02 = [
    md("""# 02 — Feature Extraction

Produces two parallel feature views of the train/val/test split:

1. **Sparse TF-IDF** (1–2 grams, sublinear TF) — feeds the classical-ML tier.
2. **Dense sentence embeddings** (`all-MiniLM-L6-v2`) — feeds BERTopic and zero-shot similarity work.

`all-MiniLM-L6-v2` is chosen over `all-mpnet-base-v2` for CPU speed; swap if a GPU is available."""),
    code("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import numpy as np
import polars as pl
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import ARTIFACTS_DIR, set_seed
set_seed()"""),
    code("""splits = {
    name: pl.read_parquet(ARTIFACTS_DIR / f"split_{name}.parquet")
    for name in ("train", "val", "test")
}
for n, s in splits.items():
    print(f"{n:5s} {s.shape}")"""),
    code("""tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
)
X_train = tfidf.fit_transform(splits["train"]["review"].to_list())
X_val   = tfidf.transform(splits["val"]["review"].to_list())
X_test  = tfidf.transform(splits["test"]["review"].to_list())
print("TF-IDF shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Vocabulary size:", len(tfidf.vocabulary_))

sparse.save_npz(ARTIFACTS_DIR / "tfidf_train.npz", X_train)
sparse.save_npz(ARTIFACTS_DIR / "tfidf_val.npz",   X_val)
sparse.save_npz(ARTIFACTS_DIR / "tfidf_test.npz",  X_test)
joblib.dump(tfidf, ARTIFACTS_DIR / "tfidf_vectorizer.joblib")"""),
    code("""# Dense embeddings. Skip if sentence-transformers is unavailable;
# notebooks 06 and 07 will fall back to recomputing on the fly.
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for name, df in splits.items():
        emb = model.encode(
            df["review"].to_list(),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(ARTIFACTS_DIR / f"emb_{name}.npy", emb)
        print(f"  emb_{name}.npy -> {emb.shape}")
except ImportError:
    print("sentence-transformers not installed; skipping dense features.")"""),
    md("""## Artifacts produced

- `tfidf_{train,val,test}.npz` + `tfidf_vectorizer.joblib`
- `emb_{train,val,test}.npy` (if sentence-transformers is installed)
"""),
]


# ---------------------------------------------------------------------------
# 03 — Lexicon Tier
# ---------------------------------------------------------------------------

NB03 = [
    md("""# 03 — Lexicon Tier

Builds the project's **multi-dimensional desire signature** from non-trainable lexicons:

- VADER compound + (pos / neg / neu) — polarity axis
- AFINN — independent polarity baseline
- (Optional) NRC EmoLex — 8 Plutchik emotions
- (Optional) NRC VAD — continuous valence / arousal / dominance

NRC lexicons require a separate download (free for research) from <https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>. Drop the txt files into `data/raw/nrc/` and the cells below will pick them up. Without them, the pipeline still works — just a smaller signature."""),
    code("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from utils import ARTIFACTS_DIR, RAW_DIR, FIGURES_DIR, save_metrics, set_seed
set_seed()
sns.set_theme(style="whitegrid")"""),
    code("""# Use the test split for honest reporting
df = pl.read_parquet(ARTIFACTS_DIR / "split_test.parquet").to_pandas()
print(df.shape)
df.head(2)"""),
    code("""import nltk
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon")
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

vader = pd.DataFrame([sid.polarity_scores(t) for t in df["review"]])
vader.columns = [f"vader_{c}" for c in vader.columns]
print(vader.head(2))"""),
    code("""from afinn import Afinn
afinn = Afinn()
df["afinn"] = [afinn.score(t) for t in df["review"]]
df["afinn_norm"] = df["afinn"] / (df["review"].str.split().str.len().clip(lower=1))
print(df[["afinn", "afinn_norm"]].describe())"""),
    code("""# Optional NRC EmoLex
NRC_EMO = RAW_DIR / "nrc" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
emo_cols = []
if NRC_EMO.exists():
    nrc = pd.read_csv(NRC_EMO, sep="\\t", header=None, names=["word", "emotion", "value"])
    nrc = nrc[nrc["value"] == 1].pivot_table(index="word", columns="emotion", values="value", fill_value=0)
    emo_cols = list(nrc.columns)
    print("Loaded NRC EmoLex:", nrc.shape, "columns:", emo_cols)

    def emo_vec(text):
        toks = text.split()
        if not toks:
            return np.zeros(len(emo_cols))
        sub = nrc.reindex(toks).fillna(0)
        return sub.mean(axis=0).values

    emo_mat = np.vstack([emo_vec(t) for t in df["review"]])
    emo_df = pd.DataFrame(emo_mat, columns=[f"emo_{c}" for c in emo_cols], index=df.index)
else:
    print(f"NRC EmoLex not found at {NRC_EMO} — skipping (still fine).")
    emo_df = pd.DataFrame(index=df.index)"""),
    code("""# Combine into the per-review desire signature
sig = pd.concat(
    [df[["label", "review"]].reset_index(drop=True),
     vader.reset_index(drop=True),
     df[["afinn", "afinn_norm"]].reset_index(drop=True),
     emo_df.reset_index(drop=True)],
    axis=1,
)
sig.head(3)"""),
    code("""# Class-conditional means heatmap: shows which signature dims separate pos/neg
numeric_cols = [c for c in sig.columns if c not in ("label", "review")]
class_means = sig.groupby("label")[numeric_cols].mean().T
class_means.columns = ["negative (0)", "positive (1)"]

fig, ax = plt.subplots(figsize=(8, max(4, len(numeric_cols) * 0.3)))
sns.heatmap(class_means, annot=True, fmt=".3f", cmap="vlag", center=0, ax=ax)
ax.set_title("Mean signature value by polarity class")
plt.tight_layout()
fig_path = FIGURES_DIR / "03_lexicon_signature_means.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved {fig_path}")"""),
    code("""# Quick lexicon baseline accuracy: predict positive iff vader_compound > 0
preds = (sig["vader_compound"] > 0).astype(int).values
acc = (preds == sig["label"].values).mean()
print(f"VADER threshold @ 0: accuracy = {acc:.4f}")

save_metrics("lexicon_vader", {"accuracy": float(acc), "threshold": 0.0})
sig.to_parquet(ARTIFACTS_DIR / "lexicon_signature_test.parquet")
print("Saved lexicon_signature_test.parquet")"""),
    md("""## Findings to record in Methodology / Results

- VADER-only accuracy on test split (above) is the **lexicon-tier baseline**. The classical-ML and transformer tiers should beat it.
- The class-mean heatmap shows which lexicon dimensions are most discriminative — keep it for the LaTeX Methodology figure.
- Without NRC, the signature is 6-dim; with NRC EmoLex, 14-dim; add NRC VAD for 17-dim."""),
]


# ---------------------------------------------------------------------------
# 04 — Classical ML Tier
# ---------------------------------------------------------------------------

NB04 = [
    md("""# 04 — Classical ML Tier

Trains three classical models on TF-IDF features and reports unified metrics.

Models: LogisticRegression → LinearSVC → LightGBM. Optional imbalanced-learn comparison via RandomUnderSampler is included. SHAP-style coefficients give a quick interpretability figure for the report."""),
    code("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import json
import numpy as np
import polars as pl
import joblib
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
)

from utils import ARTIFACTS_DIR, FIGURES_DIR, save_metrics, set_seed
set_seed()
sns.set_theme(style="whitegrid")"""),
    code("""def load_split(name):
    X = sparse.load_npz(ARTIFACTS_DIR / f"tfidf_{name}.npz")
    df = pl.read_parquet(ARTIFACTS_DIR / f"split_{name}.parquet").to_pandas()
    return X, df["label"].values

X_train, y_train = load_split("train")
X_val,   y_val   = load_split("val")
X_test,  y_test  = load_split("test")
print("Train:", X_train.shape, " Test:", X_test.shape)"""),
    code("""def evaluate(name, model, X, y, has_proba=True):
    pred = model.predict(X)
    out = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "f1_pos":   f1_score(y, pred, pos_label=1),
        "f1_neg":   f1_score(y, pred, pos_label=0),
    }
    if has_proba and hasattr(model, "predict_proba"):
        out["roc_auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return out

results = []"""),
    code("""logreg = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, solver="liblinear")
logreg.fit(X_train, y_train)
results.append(evaluate("LogReg", logreg, X_test, y_test))
print(results[-1])"""),
    code("""svc = LinearSVC(C=1.0, max_iter=2000)
svc.fit(X_train, y_train)
# LinearSVC has no predict_proba; use decision_function for AUC
y_score = svc.decision_function(X_test)
res = evaluate("LinearSVC", svc, X_test, y_test, has_proba=False)
res["roc_auc"] = roc_auc_score(y_test, y_score)
results.append(res)
print(results[-1])"""),
    code("""try:
    import lightgbm as lgb
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        n_jobs=-1,
        random_state=42,
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20)])
    results.append(evaluate("LightGBM", lgbm, X_test, y_test))
    print(results[-1])
except ImportError:
    print("lightgbm not installed; skipping.")"""),
    code("""# Confusion matrix for the best of the three
import pandas as pd
res_df = pd.DataFrame(results).set_index("model")
best_name = res_df["f1_macro"].idxmax()
print("Best model by macro-F1:", best_name)
print(res_df.round(4))

best = {"LogReg": logreg, "LinearSVC": svc, "LightGBM": locals().get("lgbm")}.get(best_name)
preds = best.predict(X_test)
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
ax.set_title(f"{best_name} confusion matrix")
ax.set_xlabel("predicted"); ax.set_ylabel("true")
plt.tight_layout()
fig_path = FIGURES_DIR / "04_classical_confusion.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.show()"""),
    code("""# Top-coefficient features for LogReg (interpretability)
import pandas as pd
tfidf = joblib.load(ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
vocab = np.array(tfidf.get_feature_names_out())
coef = logreg.coef_.ravel()
top_pos = vocab[np.argsort(coef)[-20:]][::-1]
top_neg = vocab[np.argsort(coef)[:20]]
print("Top POSITIVE features:", list(top_pos))
print("Top NEGATIVE features:", list(top_neg))

pd.DataFrame({"positive_features": top_pos, "negative_features": top_neg}).to_csv(
    ARTIFACTS_DIR / "classical_top_features.csv", index=False
)"""),
    code("""# Imbalanced-learn comparison on a downsampled training set
try:
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    logreg_rus = LogisticRegression(max_iter=1000, n_jobs=-1, solver="liblinear")
    logreg_rus.fit(X_rus, y_rus)
    print("LogReg + RandomUnderSampler:", evaluate("LogReg+RUS", logreg_rus, X_test, y_test))
except ImportError:
    print("imbalanced-learn not installed; skipping.")"""),
    code("""save_metrics("classical_ml", {"results": results, "best_model": best_name})
joblib.dump(best, ARTIFACTS_DIR / f"classical_best_{best_name}.joblib")
print("Saved metrics and best model.")"""),
    md("""## Findings to record in Methodology / Results

- Cross-model F1 / accuracy / AUC (table above) — copy directly into the LaTeX Results section.
- Confusion matrix figure: `figures/04_classical_confusion.png`.
- Top positive/negative TF-IDF features (file: `classical_top_features.csv`) — useful qualitative interpretability content for the Discussion."""),
]


# ---------------------------------------------------------------------------
# 05 — Neural Tier (DistilBERT fine-tune)
# ---------------------------------------------------------------------------

NB05 = [
    md("""# 05 — Neural Tier (DistilBERT fine-tune)

Fine-tunes `distilbert-base-uncased` on the train split via the HuggingFace `Trainer`. CPU is technically possible but very slow; a Colab T4 / Apple-Silicon MPS / any CUDA GPU is strongly recommended.

For speed, this notebook fine-tunes on a **20K subset** of the train split by default — sufficient for a baseline fine-tuned number to beat the classical tier. Bump `MAX_TRAIN` to use the full split."""),
    code("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import os
import numpy as np
import polars as pl
import torch

from utils import ARTIFACTS_DIR, save_metrics, set_seed
set_seed()

MODEL_NAME = "distilbert-base-uncased"
MAX_TRAIN  = 20_000
MAX_VAL    = 2_000
MAX_TEST   = 5_000
NUM_EPOCHS = 2
BATCH_SIZE = 16
LR         = 5e-5

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)"""),
    code("""train = pl.read_parquet(ARTIFACTS_DIR / "split_train.parquet").to_pandas().sample(MAX_TRAIN, random_state=42)
val   = pl.read_parquet(ARTIFACTS_DIR / "split_val.parquet").to_pandas().sample(MAX_VAL,   random_state=42)
test  = pl.read_parquet(ARTIFACTS_DIR / "split_test.parquet").to_pandas().sample(MAX_TEST,  random_state=42)
print(train.shape, val.shape, test.shape)"""),
    code("""from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate as hf_eval

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def to_ds(df):
    ds = Dataset.from_pandas(df[["review", "label"]].reset_index(drop=True))
    return ds.map(
        lambda b: tok(b["review"], truncation=True, padding="max_length", max_length=192),
        batched=True,
        remove_columns=["review"],
    )

ds_train = to_ds(train)
ds_val   = to_ds(val)
ds_test  = to_ds(test)"""),
    code("""acc_metric = hf_eval.load("accuracy")
f1_metric  = hf_eval.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")["f1"],
    }

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    output_dir=str(ARTIFACTS_DIR / "distilbert_runs"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LR,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=50,
    report_to="none",
    fp16=(device == "cuda"),
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    compute_metrics=compute_metrics,
)"""),
    code("""trainer.train()"""),
    code("""metrics = trainer.evaluate(ds_test)
print(metrics)
save_metrics("neural_distilbert", {**metrics, "model": MODEL_NAME, "n_train": MAX_TRAIN})
trainer.save_model(str(ARTIFACTS_DIR / "distilbert_final"))
tok.save_pretrained(str(ARTIFACTS_DIR / "distilbert_final"))"""),
    md("""## Findings to record

- Compare DistilBERT test-set accuracy / macro-F1 (above) against the classical-ML best (notebook 04). The lift is the headline neural-tier result.
- Training curve / loss is logged to `data/artifacts/distilbert_runs/`. Export a screenshot or re-plot for the LaTeX figure."""),
]


# ---------------------------------------------------------------------------
# 06 — Zero-Shot Desire Categories
# ---------------------------------------------------------------------------

NB06 = [
    md("""# 06 — Zero-Shot Desire Categories

The project's **headline original contribution**: labels reviews along the candidate desire dimensions without any supervised training, using NLI-based zero-shot classification.

Default model is `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (smaller / faster than `-large-`); swap to the large variant if a GPU with ≥16GB is available. Inference is slow on CPU — keep the sample size small (default 2,000)."""),
    code("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from utils import ARTIFACTS_DIR, FIGURES_DIR, save_metrics, set_seed
set_seed()
sns.set_theme(style="whitegrid")

ZSL_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
N_SAMPLE  = 2_000

CANDIDATES = [
    "status seeking",
    "fear of missing out",
    "genuine need",
    "social pressure",
    "hedonic impulse",
    "advertising language echo",
]"""),
    code("""df = pl.read_parquet(ARTIFACTS_DIR / "split_test.parquet").to_pandas()
df = df.sample(N_SAMPLE, random_state=42).reset_index(drop=True)
print(df.shape)"""),
    code("""from transformers import pipeline
device = 0 if torch.cuda.is_available() else -1

zsl = pipeline(
    "zero-shot-classification",
    model=ZSL_MODEL,
    device=device,
)
print("Loaded", ZSL_MODEL, "on device", device)"""),
    code("""# Multi-label (allow co-occurring categories)
out = zsl(
    df["review"].tolist(),
    candidate_labels=CANDIDATES,
    multi_label=True,
    batch_size=8,
)

scores = []
for o in out:
    row = {label: score for label, score in zip(o["labels"], o["scores"])}
    scores.append([row[c] for c in CANDIDATES])

scores_df = pd.DataFrame(scores, columns=[c.replace(" ", "_") for c in CANDIDATES])
desire = pd.concat([df[["label", "review"]].reset_index(drop=True), scores_df], axis=1)
desire.head(3)"""),
    code("""# Cross-tabulation: mean desire-category probability by polarity class
class_means = desire.groupby("label")[scores_df.columns.tolist()].mean()
print(class_means)

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(class_means, annot=True, fmt=".3f", cmap="vlag", center=0.5, ax=ax,
            yticklabels=["negative (0)", "positive (1)"])
ax.set_title("Mean zero-shot desire-category probability by polarity")
plt.tight_layout()
fig_path = FIGURES_DIR / "06_zeroshot_desire_by_polarity.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.show()"""),
    code("""# Top-k examples per category (qualitative, useful for Appendix)
top_per_cat = {}
for c in scores_df.columns:
    idx = desire[c].nlargest(3).index
    top_per_cat[c] = desire.loc[idx, ["label", c, "review"]].to_dict("records")

import json
(ARTIFACTS_DIR / "zeroshot_examples.json").write_text(json.dumps(top_per_cat, indent=2))
print("Wrote zeroshot_examples.json with top-3 per category.")"""),
    code("""desire.to_parquet(ARTIFACTS_DIR / "zeroshot_desire.parquet")
save_metrics("zeroshot_desire", {
    "model": ZSL_MODEL,
    "n_sample": N_SAMPLE,
    "candidates": CANDIDATES,
    "class_means": class_means.to_dict(),
})"""),
    md("""## Findings to record

- The class-conditional heatmap (`figures/06_zeroshot_desire_by_polarity.png`) is the project's signature figure: it shows which desire categories load on positive vs. negative reviews.
- Per-category top-3 examples (`zeroshot_examples.json`) are strong Appendix material — pick a few to embed in the Discussion as qualitative evidence.
- Validate against a hand-labeled gold set of ~50 reviews per category to report inter-rater agreement; this is the cleanest path to a human-validation paragraph in the Discussion."""),
]


# ---------------------------------------------------------------------------
# 07 — Topic Modeling (BERTopic + LDA)
# ---------------------------------------------------------------------------

NB07 = [
    md("""# 07 — Topic Modeling

BERTopic on cached sentence embeddings, with an LDA pass for comparison. KeyBERT extracts distinctive keywords per topic.

Defaults run on the test split (~20K rows) — fast on CPU. Swap to `sample_classical.parquet` for the final report."""),
    code("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from utils import ARTIFACTS_DIR, FIGURES_DIR, set_seed
set_seed()"""),
    code("""df  = pl.read_parquet(ARTIFACTS_DIR / "split_test.parquet").to_pandas()
emb_path = ARTIFACTS_DIR / "emb_test.npy"
if emb_path.exists():
    emb = np.load(emb_path)
    print("Loaded cached embeddings:", emb.shape)
else:
    print("No cached embeddings — recomputing on the fly.")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(df["review"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)"""),
    code("""from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
hdb_model  = HDBSCAN(min_cluster_size=80, metric="euclidean", prediction_data=True)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdb_model,
    calculate_probabilities=False,
    verbose=True,
)
topics, _ = topic_model.fit_transform(df["review"].tolist(), embeddings=emb)
print("Discovered", len(set(topics)), "topics (incl. noise label -1).")"""),
    code("""topic_info = topic_model.get_topic_info()
print(topic_info.head(15))
topic_info.to_csv(ARTIFACTS_DIR / "bertopic_topic_info.csv", index=False)"""),
    code("""# Top-N words per top 8 topics
top_topics = topic_info[topic_info["Topic"] != -1].head(8)["Topic"].tolist()
for t in top_topics:
    words = topic_model.get_topic(t)
    print(f"\\nTopic {t} (size {topic_info.loc[topic_info.Topic == t, 'Count'].values[0]}):")
    for w, score in words[:8]:
        print(f"  {w:20s} {score:.4f}")"""),
    code("""# BERTopic interactive viz -> static PNG via matplotlib of barchart
try:
    fig = topic_model.visualize_barchart(top_n_topics=8, n_words=8)
    fig.write_html(str(FIGURES_DIR / "07_bertopic_barchart.html"))
    print("Saved interactive barchart to figures/07_bertopic_barchart.html")
except Exception as e:
    print("BERTopic viz skipped:", e)"""),
    code("""# Per-topic mean polarity — finds topics where positive reviews concentrate
df["topic"] = topics
polar = df.groupby("topic")["label"].agg(["mean", "count"]).rename(columns={"mean": "pos_share"})
polar = polar.sort_values("count", ascending=False).head(15)
print(polar)
polar.to_csv(ARTIFACTS_DIR / "bertopic_polarity_per_topic.csv")"""),
    code("""# KeyBERT — sharper keywords than BERTopic's c-TF-IDF in some cases
try:
    from keybert import KeyBERT
    kb = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")
    samples = df.groupby("topic")["review"].apply(lambda s: " ".join(s.head(50))).head(8)
    kb_keywords = {t: kb.extract_keywords(text, top_n=8, keyphrase_ngram_range=(1, 2))
                   for t, text in samples.items()}
    for t, kws in kb_keywords.items():
        print(f"Topic {t}: {[k for k, _ in kws]}")
except ImportError:
    print("KeyBERT not installed; skipping.")"""),
    code("""# Optional: LDA comparison (Gensim) on the same texts, briefly
try:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    docs = [r.split() for r in df["review"].head(5000).tolist()]
    dct = Dictionary(docs); dct.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dct.doc2bow(d) for d in docs]
    lda = LdaModel(corpus, num_topics=8, id2word=dct, passes=2, random_state=42)
    print("\\nLDA top words per topic:")
    for i, t in lda.print_topics(num_words=6):
        print(f"  Topic {i}: {t}")
except ImportError:
    print("gensim not installed; skipping LDA comparison.")"""),
    code("""df[["label", "topic", "review"]].to_parquet(ARTIFACTS_DIR / "topics_test.parquet")
print("Saved topics_test.parquet")"""),
    md("""## Findings to record

- BERTopic topic count and a 5–8 row table of top-words-per-topic (file: `bertopic_topic_info.csv`).
- Per-topic mean polarity table (file: `bertopic_polarity_per_topic.csv`) — surfaces *which themes are most loaded with positive desire language*.
- LDA top-words for the same N topics — comparison for the Discussion."""),
]


# ---------------------------------------------------------------------------
# 08 — Evaluation & Figures
# ---------------------------------------------------------------------------

NB08 = [
    md("""# 08 — Evaluation & Figures

Compiles cross-tier metrics from `data/artifacts/metrics_*.json` into a single comparison table and figure ready for the LaTeX Results section."""),
    code("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import ARTIFACTS_DIR, FIGURES_DIR, load_all_metrics
sns.set_theme(style="whitegrid")"""),
    code("""metrics = load_all_metrics()
print("Loaded metrics:", list(metrics.keys()))"""),
    code("""rows = []

# Lexicon
if "lexicon_vader" in metrics:
    m = metrics["lexicon_vader"]
    rows.append({"tier": "lexicon", "model": "VADER@0", "accuracy": m.get("accuracy")})

# Classical
if "classical_ml" in metrics:
    for r in metrics["classical_ml"]["results"]:
        rows.append({
            "tier": "classical",
            "model": r["model"],
            "accuracy": r.get("accuracy"),
            "f1_macro": r.get("f1_macro"),
            "roc_auc":  r.get("roc_auc"),
        })

# Neural
if "neural_distilbert" in metrics:
    m = metrics["neural_distilbert"]
    rows.append({
        "tier": "neural",
        "model": m.get("model"),
        "accuracy": m.get("eval_accuracy"),
        "f1_macro": m.get("eval_f1_macro"),
    })

results = pd.DataFrame(rows)
print(results.round(4))
results.to_csv(ARTIFACTS_DIR / "final_comparison.csv", index=False)"""),
    code("""# Bar chart across tiers
plot_df = results.dropna(subset=["accuracy"]).copy()
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=plot_df, x="model", y="accuracy", hue="tier", ax=ax, dodge=False)
ax.set_ylim(0.5, 1.0); ax.set_title("Accuracy by tier and model")
ax.set_xlabel(""); plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig_path = FIGURES_DIR / "08_tier_comparison.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved {fig_path}")"""),
    code("""# Markdown summary block ready to paste into the LaTeX Results section
md_table = results.round(4).to_markdown(index=False)
out = (ARTIFACTS_DIR / "final_results_table.md")
out.write_text("## Final cross-tier results\\n\\n" + md_table + "\\n")
print(out.read_text())"""),
    md("""## What to copy into the LaTeX report

- `figures/08_tier_comparison.png` — Results section primary figure.
- `data/artifacts/final_comparison.csv` — Results table (convert to LaTeX with `pandas.to_latex` or paste the markdown).
- `figures/03_lexicon_signature_means.png`, `figures/04_classical_confusion.png`, `figures/06_zeroshot_desire_by_polarity.png` — supporting figures.
- `data/artifacts/zeroshot_examples.json` — qualitative examples for the Appendix."""),
]


# ---------------------------------------------------------------------------
# Build all
# ---------------------------------------------------------------------------

NOTEBOOKS = {
    "00_setup_and_data_audit.ipynb":         NB00,
    "01_preprocessing_and_sampling.ipynb":   NB01,
    "02_feature_extraction.ipynb":           NB02,
    "03_lexicon_tier.ipynb":                 NB03,
    "04_classical_ml_tier.ipynb":            NB04,
    "05_neural_tier_finetune.ipynb":         NB05,
    "06_zero_shot_desire_categories.ipynb":  NB06,
    "07_topic_modeling.ipynb":               NB07,
    "08_evaluation_and_figures.ipynb":       NB08,
}


def main():
    for name, cells in NOTEBOOKS.items():
        write_nb(name, cells)
    print(f"\nDone. {len(NOTEBOOKS)} notebooks written under {CODE_DIR}")


if __name__ == "__main__":
    main()
