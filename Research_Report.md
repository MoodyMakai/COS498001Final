# Research Report: Tools, Models & Datasets for a Mindful-Consumption NLP Pipeline

**Project context.** Applied NLP capstone (COS498001) following the methodology of `koosha-t/Sentiment-Analysis-NLP-for-Marketting` but inverted in purpose: instead of amplifying desire for marketing, this pipeline *detects and classifies* desire-language, FOMO, advertising echo, and impulsive-purchase signals in product reviews. Primary corpus is the Kaggle Amazon Reviews dataset (`kritanjalijain/amazon-reviews`).

**Methodology note for this report.** This research was compiled without live web access in-session. URLs below are canonical project homepages, GitHub repos, HuggingFace model cards, and dataset pages that are stable and well-established. Each link should be re-verified by the team before final LaTeX submission, and feature/version claims confirmed against current docs.

**Dataset note (read first).** The brief reports ~34.7M reviews, but the canonical `kritanjalijain/amazon-reviews` Kaggle entry is the **Zhang et al. (2015) binary polarity subset** — 3.6M train + 400K test = 4M total, with `polarity ∈ {1,2}`, `title`, `text`. The 34.7M figure refers to the larger McAuley/UCSD raw corpus (different distribution). **First-load action:** print `df.shape`, `df['polarity'].value_counts()`, and a few rows; reconcile with the brief before committing to a sampling plan. If the team needs product-category metadata (Electronics, Toys, Beauty, Fashion) for category-stratified analysis, the Kaggle binary set does *not* include it — the McAuley UCSD Amazon Reviews 2018 / 2023 release does. Plan accordingly.

---

## 1. Executive Summary

The NLP ecosystem for review-text classification is mature and almost every component this project needs already exists as a permissively licensed library or pre-trained model. The team should *integrate* aggressively and *build* sparingly — original contribution should concentrate on (a) the multi-dimensional desire signature, (b) the desire-vs-genuine-need re-labeling scheme, and (c) the contrastive comparison between Amazon-review language and minimalist-community language.

**Strongest off-the-shelf fits.** For the lexicon tier, **VADER + NRC EmoLex + NRC VAD** cover polarity, eight basic emotions, and continuous valence-arousal-dominance — directly mapping to the report's required "multi-dimensional desire signature." For the classical-ML tier, **scikit-learn** with TF-IDF and logistic regression is the established baseline of the reference repo and remains the right starting point; **LightGBM** on TF-IDF gives a stronger, still CPU-friendly classical baseline. For the neural tier, three pre-trained models stand out: **`nlptown/bert-base-multilingual-uncased-sentiment`** (fine-tuned on product reviews, predicts 1–5 stars), **`siebert/sentiment-roberta-large-english`** (single-label binary, very strong on reviews), and **`LiYuan/amazon-review-sentiment-analysis`** (Amazon-specific). For zero-shot desire-category labeling — the project's most novel contribution — **`facebook/bart-large-mnli`** and **`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`** can label reviews as "status-seeking," "FOMO-driven," "genuine-need," etc. without any supervised training.

**Topic discovery and ABSA.** **BERTopic** (with UMAP+HDBSCAN) is the consensus modern choice over plain LDA — it produces interpretable clusters from sentence embeddings and integrates cleanly with HuggingFace models. **PyABSA** provides a one-line ABSA inference path for extracting desire-triggering aspects (e.g., "fast shipping," "limited edition"). **KeyBERT** complements BERTopic with per-cluster keyword extraction.

**Supplementary data.** **GoEmotions** (28 fine-grained emotions, Reddit) is the cleanest supplementary corpus for emotion granularity. **r/minimalism / r/BuyItForLife / r/Frugal** (via Pushshift archives or HuggingFace community dumps) form the contrast corpus needed to operationalize "manufactured want vs. genuine need." **Mathur et al. dark-patterns** dataset and **Persuasion for Good** provide labeled persuasion-language signal.

**What must be built.** No public dataset labels reviews along the four desire dimensions the project proposes (impulsive-desire intensity, ad-language fingerprint, status-seeking, genuine-need). The team's original contribution is therefore the **labeling scheme itself**, implemented via (1) curated seed-lexicon expansion using sentence embeddings, (2) zero-shot bootstrapping with NLI models, and (3) a small (300–600 review) hand-annotated gold set for evaluation. This is achievable within a single capstone semester.

**Compute footprint.** The full lexicon and classical-ML tier runs on CPU in minutes on a 100K–500K stratified sample. Transformer fine-tuning on the same sample fits a single Colab T4 with **DistilBERT** in ~30–60 min; full BERT-base needs LoRA + bitsandbytes 8-bit to fit comfortably. Recommend prototyping on a 200K stratified sample first; only scale to the full Kaggle 4M after the pipeline is stable.

---

## 2. Domain-by-Domain Findings

### Domain 1 — Sentiment Analysis & Desire-Language Detection

| # | Name + Link | Category | Relevance to this pipeline | Difficulty | License |
|---|---|---|---|---|---|
| 1 | [VADER](https://github.com/cjhutto/vaderSentiment) (also in NLTK) | Lexicon-based sentiment | Core baseline; same tool the reference repo uses. Outputs compound score in [-1, +1] — directly the "sentiment polarity" axis of the desire signature. Handles negation and intensifiers. | Low | Apache-2.0 |
| 2 | [NRC Emotion Lexicon (EmoLex)](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) | Emotion lexicon (8 emotions + valence) | Adds Plutchik-8 emotion scores (anticipation, trust, joy, surprise, fear, sadness, disgust, anger) — directly relevant for separating excitement-desire from anxiety-desire. Free for research. | Low | Free for research, commercial license required |
| 3 | [NRC VAD Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html) | Continuous valence/arousal/dominance | ~20K English words scored on three continuous dimensions; ideal for a continuous "impulsive desire intensity" axis (high arousal + high valence = impulsive desire). | Low | Free for research |
| 4 | [AFINN-165](https://github.com/fnielsen/afinn) | Lexicon (integer-scored) | Simple, small, fast lexicon (-5 to +5 per word). Useful as a third lexicon for ensemble polarity scoring. | Low | Open Database License |
| 5 | [SentiWordNet 3.0](https://github.com/aesuli/SentiWordNet) | Lexicon (WordNet-based) | Adds POS-disambiguated polarity at the synset level. Useful when an ambiguous word like "sick" varies by sense. | Low | CC-BY-SA |
| 6 | [siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english) | Pre-trained transformer (binary) | Fine-tuned on diverse English review-style data; strong out-of-the-box accuracy on Amazon reviews. Good choice for the binary polarity task. | Low | MIT |
| 7 | [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) | Pre-trained transformer (1–5 stars) | Fine-tuned on product reviews (Amazon, Yelp, etc.); predicts 1–5 stars and lets you derive a continuous score. Multilingual is a free bonus. | Low | MIT |
| 8 | [LiYuan/amazon-review-sentiment-analysis](https://huggingface.co/LiYuan/amazon-review-sentiment-analysis) | Pre-trained transformer (Amazon-specific) | Explicitly fine-tuned on Amazon reviews; useful as a baseline whose performance can be reported and beaten. Verify model card for current state. | Low | check model card |
| 9 | [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | Pre-trained transformer | Trained on social-media text; underperforms on long reviews but useful as a domain-shift comparison baseline. | Low | MIT |
| 10 | [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | Emotion classifier (Ekman-6 + neutral) | Lightweight DistilRoBERTa for 7-way emotion classification. Cheap to run on the full sample; gives an emotion vector per review. | Low | Apache-2.0 |
| 11 | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) | Emotion classifier (28 labels) | Fine-grained 28-emotion model trained on GoEmotions. Captures envy, desire, admiration, excitement — directly relevant to manufactured-desire detection. | Low | MIT |
| 12 | [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) | Zero-shot classifier (NLI) | The classic zero-shot baseline. Tag reviews with custom labels ("status-seeking," "FOMO," "genuine need," "social-pressure," "hedonic-impulse") without supervised training — the project's most novel labeling layer. | Low | MIT |
| 13 | [MoritzLaurer/deberta-v3-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) | Zero-shot classifier (DeBERTa-v3) | Stronger than `bart-large-mnli` on most zero-shot benchmarks; same usage pattern via `pipeline("zero-shot-classification")`. Recommended primary zero-shot model. | Low | MIT |
| 14 | [PyABSA](https://github.com/yangheng95/PyABSA) | Aspect-Based Sentiment Analysis | One-line API for aspect extraction + aspect-level sentiment on reviews. Surfaces *which* product features drive desire (price, scarcity, packaging, status), enabling per-aspect desire scoring. | Med | MIT |
| 15 | [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) | ABSA implementations (ATAE, MemNet, IAN, BERT-SPC) | Reference implementations of classical ABSA architectures if the team wants to implement rather than just use ABSA — useful for the Methodology section. | Med | MIT |

**Synthesis.** The lexicon layer is essentially solved: VADER + NRC EmoLex + NRC VAD give you a 12-dimensional emotional signature per review with no training data. The transformer layer for binary/star sentiment is also essentially solved — three Amazon/review-tuned models are a copy-paste away. The genuine novelty for this project lives in two places: (1) the **zero-shot desire-category labels** via `deberta-v3-large-zeroshot-v2.0` (status-seeking, FOMO, genuine-need, hedonic-impulse) — there is no supervised dataset for these, but NLI-based zero-shot performs surprisingly well at ~70–80% agreement with human labels in published studies, and (2) **PyABSA** for aspect-level desire extraction, which the reference repo does not do. Together these two additions are the project's clearest claim to going beyond the koosha-t baseline.

---

### Domain 2 — Text Preprocessing & Feature Engineering

| # | Name + Link | Category | Relevance | Difficulty | License |
|---|---|---|---|---|---|
| 1 | [NLTK](https://www.nltk.org/) | NLP toolkit | Reference-repo standard. Tokenization, stopwords, stemming (Porter, Snowball), lemmatization (WordNet), VADER. Slowest of the three options at scale. | Low | Apache-2.0 |
| 2 | [spaCy](https://spacy.io/) | Industrial NLP | 10–100× faster than NLTK; pipeline-style API; built-in lemmatization, POS, NER. Recommended primary preprocessor for the full corpus. Use `en_core_web_sm` for speed, `en_core_web_lg` only if NER quality matters. | Low | MIT |
| 3 | [clean-text](https://github.com/jfilter/clean-text) | Text normalization | Handles unicode normalization, currency, emoji, URLs, phone numbers, etc. in one call. Cleaner default than rolling your own regex. | Low | MIT |
| 4 | [TextBlob](https://github.com/sloria/TextBlob) | Lightweight NLP | NLTK wrapper with a friendly API; useful for quick demos or sanity-check sentiment. | Low | MIT |
| 5 | [scikit-learn TfidfVectorizer / CountVectorizer](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) | Feature extraction (sparse) | Standard TF-IDF and BoW. Use `min_df=5, max_df=0.95, ngram_range=(1,2), sublinear_tf=True` as a strong default for review text. | Low | BSD-3 |
| 6 | [rank_bm25](https://github.com/dorianbrown/rank_bm25) | BM25 ranking | Drop-in BM25 implementation if the team wants to compare BM25 vs. TF-IDF as a stronger sparse baseline. | Low | Apache-2.0 |
| 7 | [Gensim](https://radimrehurek.com/gensim/) | Word embeddings + LDA | `Word2Vec`, `FastText` training; pre-trained vector loading; LDA. Useful for training domain-specific Word2Vec on the Amazon corpus to expand the desire-lexicon. | Low | LGPL-2.1 |
| 8 | [sentence-transformers](https://www.sbert.net/) | Sentence embeddings | The canonical library. `all-MiniLM-L6-v2` is the speed/quality default; `all-mpnet-base-v2` for higher quality; `BAAI/bge-large-en-v1.5` for SOTA. Powers BERTopic, semantic search, and lexicon expansion. | Low | Apache-2.0 |
| 9 | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | Sentence embeddings (open) | Top open-weight embedding model in the BGE family; runs locally; strong on review-style text. | Low | MIT |
| 10 | [Universal Sentence Encoder (TF Hub)](https://tfhub.dev/google/universal-sentence-encoder/4) | Sentence embeddings | Older but still useful; CPU-friendly DAN variant. Mention in Methodology as a baseline embedder. | Low | Apache-2.0 |
| 11 | [HuggingFace `datasets`](https://huggingface.co/docs/datasets) | Dataset loader | Memory-mapped Arrow backend handles the 4M Kaggle corpus comfortably; built-in streaming for the full 34M McAuley dump if scaled. Native `.map()` for preprocessing. **Recommended primary loader.** | Low | Apache-2.0 |
| 12 | [Polars](https://pola.rs/) | DataFrame (Rust) | 5–30× faster than Pandas for the CSV→DataFrame ingestion of the 4M-row Kaggle file; lazy execution avoids OOM. | Low | MIT |
| 13 | [DuckDB](https://duckdb.org/) | In-process OLAP DB | One-line CSV → SQL queryable table; ideal for category-stratified sampling and exploratory analysis without loading everything into RAM. | Low | MIT |
| 14 | [Dask](https://www.dask.org/) | Parallel Pandas | Pandas-compatible API for out-of-core processing; useful only if scaling to the full McAuley 34M+ corpus. Overkill for the 4M Kaggle set. | Med | BSD-3 |
| 15 | [Empath](https://github.com/Ejhfast/empath-client) | Lexicon expansion (legacy) | Stanford's category-builder; expands a small seed list of desire-words into a larger lexicon via Word2Vec similarity. Aging but still useful. | Low | MIT |
| 16 | [WEFE / lexicon_expansion (manual seed → embedding)](https://github.com/dccuchile/wefe) | Embedding-based lexicon expansion | Modern alternative: take 20–50 hand-picked desire seed words, expand via cosine similarity in `all-mpnet-base-v2` space to build a custom desire-lexicon. | Med | BSD-3 |

**Synthesis.** For a 4M-row corpus, the right loader is **HuggingFace `datasets`** (or **Polars** if the team prefers DataFrame ergonomics) — Pandas alone will work but uses 3–5× more memory than necessary. Use **spaCy** (not NLTK) for the actual preprocessing pass; NLTK can stay as the reference-repo-aligned default for lexicon-based sentiment but is too slow as a tokenizer at this scale. For features, run two parallel pipelines: a sparse pipeline (**TF-IDF unigrams + bigrams**) feeding the classical-ML tier, and a dense pipeline (**`all-mpnet-base-v2` sentence embeddings**) feeding BERTopic and any embedding-based lexicon expansion. Don't waste time on Word2Vec/GloVe/FastText *unless* the team explicitly wants them as baselines for the report — sentence-transformers dominate them on every modern review-text task.

---

### Domain 3 — Text Classification Models & Training Infrastructure

| # | Name + Link | Category | Relevance | Difficulty | License |
|---|---|---|---|---|---|
| 1 | [scikit-learn](https://scikit-learn.org/) | Classical ML | Reference-repo backbone. `LogisticRegression`, `MultinomialNB`, `LinearSVC`, `RandomForestClassifier`, `GradientBoostingClassifier`. Use `Pipeline` + `GridSearchCV` for clean reporting. | Low | BSD-3 |
| 2 | [LightGBM](https://lightgbm.readthedocs.io/) | Gradient boosting | Strongest classical baseline on TF-IDF features; CPU-friendly; fast. Recommended as the "best classical" comparison point. | Low | MIT |
| 3 | [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosting | Sibling to LightGBM; pick one. LightGBM is slightly faster on sparse text features. | Low | Apache-2.0 |
| 4 | [HuggingFace `transformers`](https://huggingface.co/docs/transformers) | Transformer training | `AutoModelForSequenceClassification` + `Trainer` is the canonical fine-tuning path. Native AMP, gradient accumulation, early stopping. | Low | Apache-2.0 |
| 5 | [`accelerate`](https://huggingface.co/docs/accelerate) | Distributed/AMP launcher | One-line wrapper for mixed precision and multi-GPU; useful even on a single Colab T4. | Low | Apache-2.0 |
| 6 | [`peft` (LoRA / QLoRA)](https://github.com/huggingface/peft) | Parameter-efficient fine-tuning | Fits BERT-large or DeBERTa-v3-large into a Colab T4 by training only ~0.1% of parameters. Strongly recommended for the neural tier. | Low | Apache-2.0 |
| 7 | [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) | 8/4-bit quantization | Loads transformers in INT8 for inference and 4-bit for QLoRA fine-tuning. Required for any fine-tuning above DistilBERT on free-tier GPUs. | Low | MIT |
| 8 | [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) | Lightweight transformer | 40% smaller, 60% faster than BERT-base, retains ~97% of accuracy. **Recommended primary neural model** for the project's compute budget. | Low | Apache-2.0 |
| 9 | [DeBERTa-v3-base / -small](https://huggingface.co/microsoft/deberta-v3-base) | Strong small transformer | Outperforms RoBERTa-base on most classification benchmarks at similar size. Excellent secondary neural model. | Low | MIT |
| 10 | [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) / [ALBERT](https://huggingface.co/albert/albert-base-v2) | Ultra-light transformers | Mention in Methodology for completeness; in practice DistilBERT dominates the speed/quality trade-off. | Low | Apache-2.0 |
| 11 | [`imbalanced-learn`](https://imbalanced-learn.org/) | Class imbalance | SMOTE, RandomOverSampler, RandomUnderSampler, Tomek-links. Reference-repo aligned. Note: SMOTE on *raw text* is meaningless — apply it on TF-IDF or embedding features only. | Low | MIT |
| 12 | [`focal_loss`](https://github.com/AdeelH/pytorch-multi-class-focal-loss) | Loss function | Better than naive class weighting for highly imbalanced multi-class transformer fine-tuning. | Low | MIT |
| 13 | [SetFit](https://github.com/huggingface/setfit) | Few-shot classification | Sentence-transformer + logistic head; trains in seconds on 8–64 examples per class. Ideal for the project's *hand-labeled* desire-category gold set (300–600 examples) before any zero-shot bootstrap. | Low | Apache-2.0 |
| 14 | [Skorch](https://github.com/skorch-dev/skorch) | sklearn ↔ PyTorch wrapper | Lets you put a PyTorch model inside an sklearn `Pipeline` and `GridSearchCV`; reduces boilerplate when reporting hyperparameter sweeps. | Med | BSD-3 |
| 15 | [Optuna](https://optuna.org/) | Hyperparameter optimization | Bayesian search over LR, batch size, weight decay; integrates natively with HF `Trainer`. Use for the report's "training details" section. | Low | MIT |
| 16 | [HuggingFace `evaluate`](https://huggingface.co/docs/evaluate) | Metrics library | Drop-in `accuracy`, `f1`, `precision`, `recall`, `roc_auc`; unifies metric reporting across sklearn and transformer runs. | Low | Apache-2.0 |

**Synthesis.** Mirror the reference repo's three-tier structure exactly — lexicon → classical ML → transformer — but upgrade each tier: replace VADER-only with **VADER + NRC** for the lexicon tier; replace LogReg-only with **LogReg → LinearSVC → LightGBM** as a within-tier progression; replace BERT-only with **DistilBERT (base) → DeBERTa-v3-base (LoRA) → siebert/nlptown (zero-shot inference)** as the neural tier. Use `imbalanced-learn` on the TF-IDF features for the classical tier, and class-weighted cross-entropy or focal loss for the neural tier — *do not* SMOTE text before tokenization. Track everything with **Optuna + HF `evaluate`** so the Methodology section has clean hyperparameter and metric tables.

---

### Domain 4 — Topic Modeling & Unsupervised Discovery

| # | Name + Link | Category | Relevance | Difficulty | License |
|---|---|---|---|---|---|
| 1 | [BERTopic](https://github.com/MaartenGr/BERTopic) | Neural topic model | **Recommended primary topic model.** Combines sentence embeddings + UMAP + HDBSCAN + class-based TF-IDF for coherent, interpretable topics. Native `topics_per_class` for per-product-category topic comparison. | Low | MIT |
| 2 | [Gensim LDA](https://radimrehurek.com/gensim/models/ldamodel.html) | Probabilistic topic model | Classical baseline. Run alongside BERTopic for comparison in the Results section — LDA's topics will be noticeably worse, which itself is a finding worth reporting. | Low | LGPL-2.1 |
| 3 | [scikit-learn LatentDirichletAllocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) | LDA (alternative impl.) | Drop-in alternative to Gensim LDA inside an sklearn Pipeline. | Low | BSD-3 |
| 4 | [Top2Vec](https://github.com/ddangelov/Top2Vec) | Neural topic model (alt) | Doc2Vec-based; older than BERTopic but sometimes produces complementary clusters. Worth a comparison run if time allows. | Low | BSD-3 |
| 5 | [Contextualized Topic Models (CTM)](https://github.com/MilaNLProc/contextualized-topic-models) | Neural topic model (alt) | Combines BERT embeddings with a neural LDA-like generative model. More complex to set up than BERTopic; not recommended unless the team wants to compare. | Med | MIT |
| 6 | [UMAP](https://umap-learn.readthedocs.io/) | Dimensionality reduction | The de facto pre-clustering reducer for sentence embeddings. Used internally by BERTopic. Tune `n_neighbors=15, min_dist=0.0, metric='cosine'`. | Low | BSD-3 |
| 7 | [HDBSCAN](https://hdbscan.readthedocs.io/) | Density clustering | Companion to UMAP; produces variable-density clusters and a noise label. Used internally by BERTopic. | Low | BSD-3 |
| 8 | [pyLDAvis](https://github.com/bmabey/pyLDAvis) | LDA visualization | Interactive topic visualization (PCA + relevance metric). Great for the Appendix. | Low | BSD-3 |
| 9 | [KeyBERT](https://github.com/MaartenGr/KeyBERT) | Keyword extraction (BERT) | Same author as BERTopic. Extracts the most distinctive keywords per cluster using cosine similarity. Use for surfacing per-topic desire-vocabulary. | Low | MIT |
| 10 | [YAKE](https://github.com/LIAAD/yake) | Keyword extraction (statistical) | Unsupervised, language-agnostic, no embedding model needed. CPU-friendly baseline. | Low | GPL-3.0 (note: copyleft) |
| 11 | [RAKE-NLTK](https://github.com/csurfer/rake-nltk) | Keyword extraction (statistical) | Even simpler than YAKE; useful as a third keyword-extraction baseline. | Low | MIT |

**Synthesis.** **BERTopic** is the only modern choice that needs to be in the pipeline — it subsumes UMAP, HDBSCAN, and class-based TF-IDF, integrates with HuggingFace embeddings, and produces interpretable topics out of the box. Run a Gensim LDA pass alongside it purely as a "before vs. after" baseline for the Discussion section: showing that BERTopic produces semantically tighter clusters than LDA on review text is a clean Methodology contribution. Layer **KeyBERT** on top to extract the top-N desire-words per cluster — these become the per-topic vocabulary tables in Results. Visualize with `BERTopic.visualize_topics()` and `pyLDAvis` (the latter for the Appendix).

---

### Domain 5 — Datasets for Supplementary Training & Evaluation

| # | Name + Link | Category | Relevance | Difficulty | License |
|---|---|---|---|---|---|
| 1 | [Kaggle: kritanjalijain/amazon-reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) | Primary corpus | Project's main dataset. Verify on first load: expected ~4M rows (Zhang 2015 binary polarity), `polarity ∈ {1,2}`, `title`, `text`. No category metadata. | — | Per Kaggle dataset terms |
| 2 | [McAuley UCSD Amazon Reviews 2018](https://nijianmo.github.io/amazon/index.html) | Larger Amazon corpus (alt) | If category-stratified analysis (Electronics, Toys, Beauty, Fashion) is needed, switch to the 5-core per-category subsets here. ~233M reviews total; per-category subsets are 1–10M each. | Med | Per-publisher terms (research use) |
| 3 | [McAuley Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) | Updated Amazon corpus | 2023 refresh of the McAuley dataset; reviews up to Sep 2023. Recommended over 2018 if available, for currency in the Discussion. | Med | Per-publisher terms |
| 4 | [GoEmotions (Google)](https://github.com/google-research/google-research/tree/master/goemotions) | Emotion-labeled corpus | 58K Reddit comments, 28 emotion labels. **Best supplementary corpus** for fine-tuning or transfer-learning a fine-grained emotion model that can then be applied to Amazon reviews. | Low | Apache-2.0 |
| 5 | [SemEval-2018 Task 1: Affect in Tweets](https://competitions.codalab.org/competitions/17751) | Emotion + V/A/D | E-c (multi-label emotion), EI-reg (emotion intensity), V-reg (valence regression). Useful as evaluation transfer for the desire-intensity axis. | Med | Research use |
| 6 | [EmoBank](https://github.com/JULIELab/EmoBank) | Continuous V/A/D | 10K English sentences with continuous valence/arousal/dominance. Pairs with NRC-VAD for the impulsive-desire axis. | Low | CC-BY-SA |
| 7 | [Yelp Open Dataset](https://www.yelp.com/dataset) | Review corpus (cross-domain) | Useful for reporting cross-domain transfer performance: train on Amazon, test on Yelp (or vice versa). Strong Methodology figure. | Med | Yelp dataset license (research use) |
| 8 | [SST-2 / SST-5 (Stanford Sentiment Treebank)](https://nlp.stanford.edu/sentiment/) | Sentiment benchmark | Classic short-text sentiment benchmark; useful as a sanity-check for the lexicon and classical-ML tiers. | Low | Research use |
| 9 | [IMDB Movie Reviews (Maas et al. 2011)](https://ai.stanford.edu/~amaas/data/sentiment/) | Long-review corpus | The canonical long-review sentiment dataset. Use for cross-domain comparison or as a smaller initial corpus while the team is debugging the pipeline. | Low | Research use |
| 10 | [Mathur et al. — Dark Patterns (2019)](https://webtransparency.cs.princeton.edu/dark-patterns/) | Persuasive UX corpus | ~1,800 dark-pattern instances scraped from 11K shopping sites with a 7-category taxonomy. Excellent supplementary signal for "advertising-language fingerprint." | Med | Per-publisher terms |
| 11 | [Persuasion for Good (Wang et al. 2019)](https://convokit.cornell.edu/documentation/persuasionforgood.html) | Persuasion-language corpus | ~1K dialogues annotated with 10 persuasion strategies (logical appeal, emotion appeal, credibility, scarcity, etc.). Maps directly onto the project's "manufactured desire" categories. | Med | Per-publisher terms |
| 12 | [Reddit r/minimalism, r/BuyItForLife, r/Frugal (Pushshift archives via HF Hub)](https://huggingface.co/datasets) | Contrast corpus | The single most important *contrast* dataset for the project: minimalism-community language as a foil to consumerist Amazon-review language. Search HF Hub for community-curated dumps; otherwise scrape via PRAW or the Pushshift archive (note: Pushshift access policy changed in 2023 — verify current availability). | Med | Reddit content terms; verify any HF dump license |
| 13 | [AnnoMI](https://github.com/uccollab/AnnoMI) | Motivational-interviewing dialogue | Annotated MI counseling sessions; outside the core review-classification scope but relevant if the team wants a Discussion paragraph on "what compassionate language sounds like." | Low | CC-BY-NC-4.0 |
| 14 | [EmpatheticDialogues (Facebook AI Research)](https://github.com/facebookresearch/EmpatheticDialogues) | Empathy corpus | 25K conversations grounded in 32 emotion situations. Same caveat as AnnoMI — peripheral to the classification pipeline but useful for Discussion. | Low | CC-BY-NC |
| 15 | [Moral Foundations Twitter Corpus (MFTC)](https://osf.io/k5n7y/) | Values-language corpus | Tweets annotated with five moral foundations. A reach for this project, but interesting for the zero-shot label set's "values-based categories." | Med | Per-publisher (academic) |

**Synthesis.** The core project doesn't need anything beyond the Kaggle Amazon corpus for the polarity-classification task. Supplementary datasets matter for two specific extensions: **GoEmotions** is the highest-leverage addition (clean license, easy to integrate, directly usable to fine-tune the emotion classifier the project applies to reviews), and a **minimalism Reddit corpus** is the highest-novelty addition (no existing study contrasts Amazon-review language with minimalism-community language at scale — that's a genuinely original Discussion contribution). The Mathur dark-patterns dataset and the Persuasion for Good dataset are useful in the Discussion as *external taxonomies* the team's discovered topics can be mapped against, even if not used as training signal. **Action item:** verify the Kaggle dataset's actual size and schema *before* writing any of the LaTeX Dataset section — the brief and the canonical Kaggle entry disagree.

#### Recommended sampling plan for the Amazon Reviews corpus

The Kaggle binary polarity set is balanced 50/50 by construction (Zhang et al. 2015). For an MVP:

- **Stage 1 (debugging, 100K rows):** Random sample of 100K from the 4M Kaggle set, stratified on `polarity`. Runs the entire lexicon + classical-ML pipeline in <5 min on CPU.
- **Stage 2 (full classical-ML report, 500K rows):** Stratified random sample of 500K. Sufficient for stable F1/AUC numbers and for SMOTE experiments. Fits in memory; CPU-only.
- **Stage 3 (transformer fine-tuning, 200K rows):** Stratified random sample of 200K, with an 80/10/10 train/val/test split. Fits a DistilBERT fine-tune on a free-tier T4 in ~30–60 min. Use LoRA + 4-bit if going to BERT-base.
- **Stage 4 (final reporting, full 4M):** Lexicon-only and pre-trained-model inference on the full 4M corpus is feasible in batched mode. Fine-tuning is not — report fine-tuned numbers on the 200K stratified split.
- **If using McAuley 2023 instead** for category metadata: select **Electronics, Beauty, Toys & Games, Clothing/Shoes/Jewelry, Fashion** as the consumerist-relevant categories. Take 100K per category (500K total), stratified by 1–5 star rating, for a category-comparison Results table.
- **Loading strategy:** `datasets.load_dataset("csv", data_files=..., streaming=True)` for the first pass to characterize the corpus; **Polars** `read_csv` for the actual sampling step (fastest); persist the sampled subset as a parquet file and never re-load the raw CSV again.

---

### Domain 6 — Evaluation, Visualization & Reporting Tools

| # | Name + Link | Category | Relevance | Difficulty | License |
|---|---|---|---|---|---|
| 1 | [scikit-learn metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html) | Classification metrics | Reference-repo standard. `accuracy_score`, `f1_score`, `precision_recall_fscore_support`, `confusion_matrix`, `roc_auc_score`, `classification_report`. | Low | BSD-3 |
| 2 | [HuggingFace `evaluate`](https://huggingface.co/docs/evaluate) | Unified metrics | Wraps sklearn metrics behind a consistent API; integrates natively with `Trainer.compute_metrics`. Recommended for the neural tier. | Low | Apache-2.0 |
| 3 | [`seqeval`](https://github.com/chakki-works/seqeval) | NER/sequence metrics | Required only if the team adds a NER stage to identify product/brand mentions. | Low | MIT |
| 4 | [CheckList (Ribeiro et al. 2020)](https://github.com/marcotcr/checklist) | Behavioral testing | Test models with templated perturbations (negation, typos, name swaps). Strong addition to the Discussion: shows whether the model generalizes or merely memorizes review-specific cues. | Med | MIT |
| 5 | [SHAP](https://github.com/shap/shap) | Feature attribution | Per-prediction Shapley values; `shap.plots.text` is the cleanest way to visualize *why* a transformer flagged a review as high-desire. Strong figure for Results/Discussion. | Med | MIT |
| 6 | [LIME](https://github.com/marcotcr/lime) | Local explanations | Older, simpler alternative to SHAP; faster on classical-ML models. Pick one. | Low | BSD-2 |
| 7 | [Captum](https://captum.ai/) | PyTorch interpretability | Native integrated-gradients and attention attribution for transformers. Use if SHAP is too slow. | Med | BSD-3 |
| 8 | [WordCloud](https://github.com/amueller/word_cloud) | Visualization | Standard per-class word clouds. Decorative but expected in a report's Appendix. | Low | MIT |
| 9 | [BERTViz](https://github.com/jessevig/bertviz) | Attention visualization | Interactive transformer-attention visualizer; one notebook cell produces a publication-grade figure for the Methodology section. | Low | Apache-2.0 |
| 10 | [displaCy (spaCy)](https://spacy.io/usage/visualizers) | NER/dependency viz | Use if the team adds a brand/product NER pass. | Low | MIT |
| 11 | [seaborn / matplotlib](https://seaborn.pydata.org/) | Plotting | Standard. `seaborn.heatmap` for confusion matrices, `seaborn.barplot` for per-class F1. | Low | BSD-3 |
| 12 | [Plotly](https://plotly.com/python/) | Interactive plotting | Useful for BERTopic's interactive topic-distance maps in HTML reports; static export to PNG/SVG for LaTeX inclusion. | Low | MIT |
| 13 | [Weights & Biases (`wandb`)](https://wandb.ai/) | Experiment tracking | Free for academic use; free tier sufficient. Logs hyperparameters, metrics, learning curves, system stats. **Recommended primary tracker.** | Low | Free academic; commercial tiers exist |
| 14 | [MLflow](https://mlflow.org/) | Experiment tracking (alt) | Self-hosted alternative if W&B's hosted model is undesirable. More setup; less polished UI. | Med | Apache-2.0 |
| 15 | [TensorBoard](https://www.tensorflow.org/tensorboard) | Experiment tracking (alt) | Built into HuggingFace `Trainer` via `report_to="tensorboard"`. Good enough for a single-team capstone. | Low | Apache-2.0 |
| 16 | [`nbconvert`](https://nbconvert.readthedocs.io/) | Notebook export | Convert notebooks to HTML/PDF for the Google Drive submission. | Low | BSD-3 |
| 17 | [Jupyter Book](https://jupyterbook.org/) | Reproducible reports | Compiles a directory of notebooks into a navigable HTML book; overkill for a 6-notebook capstone but professional-looking. | Med | BSD-3 |
| 18 | [`papermill`](https://github.com/nteract/papermill) | Notebook parameterization | Run notebooks programmatically with parameter sweeps; useful for the Methodology section's hyperparameter table. | Low | BSD-3 |
| 19 | [`nbdime`](https://github.com/jupyter/nbdime) | Notebook diff/merge | Critical for multi-author git collaboration on shared notebooks. Install before the team's first merge conflict, not after. | Low | BSD-3 |

**Synthesis.** Standard sklearn metrics handle 90% of the evaluation needs and align with the reference repo. The two highest-leverage additions are **CheckList** (for the Discussion section's "what worked vs. what didn't" — perturbation testing exposes shortcut learning that raw F1 hides) and **SHAP** with `shap.plots.text` (for one or two strong qualitative figures showing *which review words* drove a high-desire prediction). Track all runs in **W&B** from day one — retroactive tracking is impossible and the academic free tier is generous. For the LaTeX paper, export figures as SVG or 300dpi PNG; for the Google Drive notebook submission, run `nbconvert` on each finished notebook as an HTML companion.

---

## 3. Recommended Pipeline Stack

| Stage | Recommended tool(s) | Notes |
|---|---|---|
| **(a) Data loading & sampling** | `datasets` (HF) + Polars + DuckDB | Stream the raw Kaggle CSV via HF `datasets`; sample with Polars; persist as parquet. DuckDB for ad-hoc category counts. |
| **(b) Preprocessing** | spaCy (`en_core_web_sm`) + `clean-text` + NLTK (VADER, stopwords for parity with reference repo) | spaCy for tokenization/lemma; NLTK kept for VADER and report-parity. |
| **(c) Feature extraction (sparse)** | scikit-learn `TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.95, sublinear_tf=True)` | Reference-repo aligned; strong default. |
| **(c) Feature extraction (dense)** | `sentence-transformers/all-mpnet-base-v2` (or `BAAI/bge-large-en-v1.5` for higher quality) | Powers BERTopic, lexicon expansion, and semantic-similarity analyses. |
| **(d) Lexicon tier** | VADER + NRC EmoLex + NRC VAD + AFINN | Produces the 12-dim emotional signature: 1 polarity + 8 emotions + 3 V/A/D. |
| **(d) Classical-ML tier** | scikit-learn `LogisticRegression` → `LinearSVC` → LightGBM, all on TF-IDF; `imbalanced-learn` for SMOTE on features | Three models = clean ablation table for Methodology/Results. |
| **(e) Neural tier (supervised polarity)** | DistilBERT fine-tune (full); DeBERTa-v3-base with LoRA via `peft`+`bitsandbytes` (stretch) | Report DistilBERT as the deliverable; DeBERTa as the upper-bound. |
| **(e) Neural tier (zero-shot desire categories)** | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` for ["status-seeking","FOMO","genuine-need","social-pressure","hedonic-impulse","ad-language-echo"] | No training data needed; gives the project's most novel labels. |
| **(e) Neural tier (emotion granularity)** | `SamLowe/roberta-base-go_emotions` (28 emotions) — inference only | Inference-only on the sampled corpus. |
| **(e) Neural tier (ABSA)** | PyABSA `ATEPC` checkpoint | Aspect extraction + per-aspect sentiment in one pass. |
| **(f) Topic modeling** | BERTopic (with `all-mpnet-base-v2` embeddings, UMAP, HDBSCAN, c-TF-IDF) + KeyBERT for per-cluster keywords; Gensim LDA as comparison baseline | BERTopic primary; LDA only for comparison. |
| **(g) Evaluation & visualization** | sklearn metrics + HF `evaluate` + CheckList + SHAP; figures via seaborn/matplotlib + BERTopic visualizers | Track every run in **W&B**. |
| **(g) Reporting** | Jupyter + `nbconvert` + `papermill` for parameterized re-runs | Mirrors reference-repo notebook structure. |

**Multi-dimensional desire signature (per review).** The above stack produces, per review, a vector of:

1. `polarity` ∈ [-1, +1] — VADER compound, plus transformer p(positive)
2. `arousal`, `valence`, `dominance` ∈ [0, 1] — NRC VAD lexicon means
3. 8 NRC emotion intensities — anticipation, joy, trust, surprise, fear, sadness, disgust, anger
4. 28 GoEmotions probabilities — fine-grained
5. 6 zero-shot desire-category probabilities — status / FOMO / genuine-need / social-pressure / hedonic-impulse / ad-echo
6. K aspect-sentiment pairs from PyABSA
7. Top-N BERTopic topic membership

This is the project's headline output and the basis for every cross-tabulation in Results.

---

## 4. Notebook Structure Recommendation

Mirroring the reference repo's "one notebook per stage" pattern:

| # | Notebook | Purpose | Tools used |
|---|---|---|---|
| `00_setup_and_data_audit.ipynb` | Environment check + data sanity | Verify Kaggle dataset shape/schema; print polarity distribution, length distribution, sample reviews. **Run before anything else.** | `datasets`, Polars, Pandas, seaborn |
| `01_preprocessing_and_sampling.ipynb` | Cleaning + stratified sampling | Build the 100K / 500K / 200K sampled splits as parquet artifacts. | spaCy, clean-text, Polars, scikit-learn `train_test_split` |
| `02_feature_extraction.ipynb` | TF-IDF + sentence embeddings | Produce `(X_tfidf, X_dense)` and persist them. | scikit-learn TfidfVectorizer, sentence-transformers |
| `03_lexicon_tier.ipynb` | Lexicon-based signature | VADER + NRC EmoLex + NRC VAD + AFINN; build the 12-dim emotional signature; per-class summary stats. | NLTK/VADER, NRC lexicons (manual download) |
| `04_classical_ml_tier.ipynb` | LogReg → LinearSVC → LightGBM | Train/eval all three on TF-IDF; SMOTE comparison; SHAP feature importance. | scikit-learn, imbalanced-learn, LightGBM, SHAP |
| `05_neural_tier_finetune.ipynb` | DistilBERT fine-tune | HF `Trainer` on the 200K split; W&B tracking; SHAP-text on a few examples. | transformers, accelerate, peft, evaluate, wandb, SHAP |
| `06_zero_shot_desire_categories.ipynb` | Zero-shot labeling + emotion + ABSA inference | Inference-only: zero-shot DeBERTa, GoEmotions, PyABSA. Persist labels. | transformers `pipeline`, PyABSA |
| `07_topic_modeling.ipynb` | BERTopic + LDA comparison | BERTopic + KeyBERT; Gensim LDA; visualization. | BERTopic, KeyBERT, Gensim, pyLDAvis |
| `08_evaluation_and_figures.ipynb` | Final metrics + LaTeX-ready figures | Compile results across all tiers; export SVG/PNG figures for the LaTeX report. | sklearn, seaborn, matplotlib, plotly |
| `09_appendix_error_analysis.ipynb` | (Optional) CheckList + manual review | Behavioral tests on the best model; small qualitative tables. | CheckList, manual analysis |

Each notebook should: (1) start with a fixed `random_state = 42`, (2) load the previous stage's parquet artifact, (3) produce a single artifact for the next stage, (4) log to W&B if any model is trained, (5) end with a markdown cell summarizing what to copy into the LaTeX report.

---

## 5. Gaps & Open Questions

1. **No supervised data exists for the desire-category labels.** "Status-seeking," "FOMO-driven," "genuine-need," "ad-echo," "social-pressure," "hedonic-impulse" — none of these have public labeled review datasets. **Bridge:** zero-shot NLI labeling with `deberta-v3-large-zeroshot-v2.0`, validated against a hand-labeled gold set of 300–600 reviews. This becomes the project's headline original contribution.
2. **The Kaggle binary polarity dataset has no product-category metadata.** This blocks any per-category analysis (Electronics vs. Toys vs. Beauty). **Bridge:** if category analysis is essential, switch to McAuley 2018/2023 5-core subsets. This is a substantive scope decision; resolve in the first week.
3. **The reference repo's three-tier structure assumes binary sentiment.** Multi-label desire categorization needs a different evaluation framing (per-label F1, macro-F1, label cardinality). Decide early whether desire categories are mutually exclusive (single-label) or co-occurring (multi-label). Recommend multi-label.
4. **No standard "manufactured-vs-genuine" benchmark exists.** Even with the contrast corpus from r/minimalism etc., there is no human-validated benchmark separating manufactured wants from genuine needs in product-review text. **Bridge:** the team's hand-labeled gold set serves dual duty as both validation and a small published artifact.
5. **Compute budget.** Free-tier Colab T4 fits DistilBERT comfortably but BERT-base requires LoRA or 4-bit. **Bridge:** plan for this from the start; do not assume a full-precision BERT-base fine-tune fits — it does not without `peft` + `bitsandbytes`.
6. **Pushshift Reddit access** has been intermittent since 2023. **Bridge:** prefer existing HF Hub dumps of r/minimalism etc. over scraping; if absolutely necessary, use PRAW with proper rate-limiting.
7. **Lexicon licensing.** NRC EmoLex is free for *research* but requires a commercial license otherwise. Document this in the report; not a blocker for academic use.
8. **Prompt-cached inference vs. transformer fine-tuning** are different evaluation regimes. The project should clearly separate "what an off-the-shelf zero-shot model says" from "what a fine-tuned model says" in the Results section to avoid conflating the two.

---

## 6. Top 3 Quick Wins (≤1 week to a credible early demo)

**Quick Win 1 — Lexicon-only desire signature on 100K reviews (1–2 days, CPU only).**
Run VADER + NRC EmoLex + NRC VAD over a 100K stratified Kaggle sample; build the 12-dimensional emotional signature; produce per-polarity-class summary tables and a heatmap. Zero training, zero GPU, fully reproducible. This is the project's first concrete deliverable and aligns with the reference repo's lexicon tier.

**Quick Win 2 — Zero-shot desire labels on the same 100K sample (1–2 days, GPU recommended).**
Apply `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` with the candidate label set ["status-seeking", "FOMO", "genuine-need", "social-pressure", "hedonic-impulse", "ad-language-echo"]. Cross-tab with star rating to produce the project's headline finding: *which star-rating bands are most loaded with which desire categories*. This is the differentiated contribution beyond the koosha-t baseline and requires no fine-tuning.

**Quick Win 3 — BERTopic over `all-mpnet-base-v2` embeddings (1 day after embeddings are computed).**
Run BERTopic with default settings on the 100K sample; output topic-keywords, topic-distance map, and per-topic mean polarity. Use KeyBERT for top-N words per topic. This produces the unsupervised discovery section's headline figures and seeds the discussion of *which product features drive which kinds of desire*.

Together these three deliverables produce a credible mid-semester milestone slide deck — three figures, two tables — without any model fine-tuning or labeled data. Fine-tuning (DistilBERT) and the LaTeX report can layer on top in the second half.

---

## Appendix A — Reference-repo alignment crosswalk

| Reference repo stage | This project equivalent | Notes |
|---|---|---|
| Data loading (Pandas) | `00_setup_and_data_audit.ipynb` + `01_preprocessing_and_sampling.ipynb` | Polars/HF datasets, but Pandas-compatible API. |
| Preprocessing (NLTK) | `01_preprocessing_and_sampling.ipynb` | spaCy primary; NLTK retained for VADER. |
| TF-IDF / BoW | `02_feature_extraction.ipynb` | Same; add sentence embeddings. |
| VADER lexicon sentiment | `03_lexicon_tier.ipynb` | Add NRC EmoLex, NRC VAD, AFINN. |
| sklearn LogReg / NB / RF / SVM | `04_classical_ml_tier.ipynb` | Same; add LightGBM. |
| BERT/transformer fine-tune | `05_neural_tier_finetune.ipynb` | DistilBERT primary; DeBERTa-v3 stretch. |
| imbalanced-learn (SMOTE) | `04_classical_ml_tier.ipynb` | Same; only on TF-IDF features. |
| Evaluation (sklearn metrics) | `08_evaluation_and_figures.ipynb` | Same; add CheckList and SHAP. |
| (no equivalent) | `06_zero_shot_desire_categories.ipynb` | **New: project's headline contribution.** |
| (no equivalent) | `07_topic_modeling.ipynb` | **New: BERTopic + LDA comparison.** |

---

## Appendix B — Caveats and verification list

Before relying on any link, model, or dataset in this report, verify:

1. **Kaggle dataset shape and schema.** Brief says 34.7M; canonical kritanjalijain entry is the Zhang 2015 4M binary set. Reconcile on first load.
2. **Pre-trained model availability.** HuggingFace model cards occasionally go private or are renamed; confirm each linked model is still public and CPU/GPU-compatible.
3. **License compatibility for the LaTeX submission.** NRC EmoLex/VAD are free for research but not commercial; CC-BY-NC datasets (EmpatheticDialogues, AnnoMI) cannot be redistributed in a public Google Drive submission — cite, don't redistribute.
4. **Pushshift / Reddit data access.** API and archive availability has shifted in 2023–2024; confirm current state before depending on r/minimalism scrapes.
5. **Compute environment.** All recommended fine-tuning workflows assume Colab T4 or equivalent. Confirm GPU access before scoping the neural-tier deliverables.
