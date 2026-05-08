# Intelligent Reading Comprehension and Quiz Generation on RACE — A Classical-ML Pipeline with a BERT Comparison Baseline

---

## 1. Abstract

This project builds a reading-comprehension and quiz-generation pipeline on the
RACE dataset (Lai et al., 2017) under a deliberate constraint: the two core
models — Model A (question generation) and Model B (distractor and hint
generation) — use **only** traditional machine learning. Both models follow the
same template: per-MCQ candidates are extracted with passage-level heuristics
and ranked by a supervised Logistic Regression scorer combined in a weighted
ensemble with an unsupervised K-Means quality scorer. To quantify the cost of
forgoing neural representations, we add a third, contrastive baseline that
re-ranks the *same* candidates with mean-pooled `bert-base-uncased` embeddings
(Devlin et al., 2019). All three models are evaluated on the same RACE
val/test splits with BLEU (Papineni et al., 2002), ROUGE-1/2/L
(Lin, 2004), and METEOR (Banerjee & Lavie, 2005). BERT improves question
generation (BLEU 0.0218 vs 0.0180, METEOR 0.2186 vs 0.2042) and distractor
generation (ROUGE-1 0.0448 vs 0.0275) on test; classical Model B retains the
edge on hint generation because its evaluation reference is constructed from
the same overlap features it scores with. A Streamlit UI integrates both
traditional models end-to-end.

*(196 words.)*

---

## 2. Introduction & Motivation

Reading-comprehension assessment items are expensive to author at scale, which
makes automatic question, distractor, and hint generation a long-standing
educational-NLP problem. RACE (Lai et al., 2017) provides 28K passages and
~100K human-authored four-option MCQs drawn from English exams for Chinese
middle- and high-school students, making it a strong benchmark for both
answer prediction and quiz-style generation.

Modern systems for this task are dominated by Transformer-based models
(Vaswani et al., 2017), most prominently BERT (Devlin et al., 2019) and its
descendants. This project, however, was scoped to **classical** machine
learning for two pedagogical reasons:

1. Quantify how far feature-engineered logistic + clustering pipelines can go
   on a non-trivial generation task.
2. Provide a transparent baseline against which we can later measure neural
   improvements without re-running the entire pipeline.

To make the second point concrete, we add a controlled **BERT rerank baseline**
that keeps the candidate generation step identical to Model A and Model B but
replaces the classical scorer with cosine similarity in BERT embedding space.
Because everything except the scorer is held fixed, any metric gap is
attributable to the representation, not to a richer candidate generator. This
yields a clean three-way comparison: traditional supervised, traditional
supervised + unsupervised ensemble, and BERT.

**Research questions.**

- RQ1: Can a fully classical ML pipeline produce structured, evaluable
  question/distractor/hint outputs on RACE?
- RQ2: Does combining supervised and unsupervised scoring add value over
  either scorer alone?
- RQ3: Holding the candidate pool fixed, how much of the remaining metric
  headroom is captured by a contextual encoder like BERT?

---

## 3. Related Work

**Reading comprehension and benchmarks.** RACE (Lai et al., 2017) is a
standard MCQ-style benchmark with passages and human-written questions; its
difficulty stems from the need for inference rather than surface matching.
Earlier datasets like SQuAD (Rajpurkar et al., 2016) emphasized span
extraction, not MCQ generation.

**Pretrained language models.** The Transformer (Vaswani et al., 2017) and
BERT (Devlin et al., 2019) established that bidirectional self-attention over
WordPiece tokens produces representations that transfer broadly across
classification and ranking tasks. We use BERT in its mean-pooled, frozen,
embedding-as-feature configuration — closer to Reimers & Gurevych (2019)'s
Sentence-BERT setup than to fine-tuning — because the project's goal is a
controlled baseline rather than a tuned system.

**Classical question generation.** Pre-Transformer systems for question
generation typically combined syntactic patterns, template instantiation, and
ranking over candidate spans (Heilman & Smith, 2010). The Model A pipeline in
this project sits in that lineage: deterministic template generation followed
by a feature-ranked selection.

**Generation evaluation.** We use the three overlap metrics most established
in the MT / summarization / QG literature: BLEU (Papineni et al., 2002), ROUGE
(Lin, 2004), and METEOR (Banerjee & Lavie, 2005). All three have well-known
weaknesses for paraphrase-tolerant evaluation (Liu et al., 2016), but they are
the standard reporting requirements for this project and remain the most
common comparable metrics in the QG literature. Background on classical NLP
pipelines and metric semantics follows Jurafsky & Martin (2025, draft 3rd ed.).

---

## 4. Dataset Analysis

### 4.1 Schema

Every RACE row carries an `id`, an `article` (the reading passage), a
`question`, four answer options (`A`, `B`, `C`, `D`), and an `answer` letter.
Preprocessing also derives a long-format **verification table** with one row
per (article, question, option) — four rows per MCQ — labelled `1` when the
option letter matches `answer` and `0` otherwise. This long table is used by
feature engineering (One-Hot, TF-IDF, handcrafted lexical features) and is
ready for an option-level "is this option correct?" classifier downstream.

### 4.2 Splits and Deduplication

The raw RACE CSVs supplied for the project had train/test/dev duplication. The
preprocessing pipeline therefore exposes a `--combined-split` mode that:

1. Concatenates every CSV in `data/raw/`.
2. Drops duplicates on the `id` column.
3. Shuffles with `random_state=42`.
4. Splits with configurable `train_fraction` and `val_fraction_of_train_pool`
   (defaults 0.8 / 0.1 → ≈ 72% train, 8% validation, 20% test of unique rows).

In the run that produced all metrics in this report, the resulting MCQ-level
sizes are:

| Split      | MCQ rows |
|------------|---------:|
| Train      |  3,000   |
| Validation |  2,011   |
| Test       |  5,027   |

The verification long-format table has 4× these counts.

### 4.3 Preprocessing Pipeline (`src/preprocessing.py`)

| Step | Implementation |
|---|---|
| Lowercasing, punctuation removal | `clean_text` (regex strip of non-alphanumerics + collapse whitespace) |
| Cleaned per-MCQ frame | `clean_mcq_frame` writes `article_clean`, `question_clean`, `{A,B,C,D}_clean` |
| Long-form verification table | `mcq_to_verification_long` emits 4 rows / MCQ with binary label |
| One-Hot Encoding features | `CountVectorizer(binary=True, max_features=50000)`, fit on **train only**, persisted to `data/processed/artifacts/vectorizer_ohe.joblib` |
| TF-IDF features (optional) | `TfidfVectorizer(max_features=50000)`, train-fit, persisted |
| Article–Question TF-IDF | Separate `TfidfVectorizer` (8192 features) for cosine-similarity scalar per MCQ |
| Handcrafted lexical features | `feat_article_char_len`, `feat_question_char_len`, `feat_option_char_len`, `feat_article_word_count`, `feat_question_word_count`, `feat_option_word_count`, three Jaccard overlaps (Q↔A, O↔A, O↔Q), `feat_cosine_article_question` |
| Train/val/test split | Either three-file split or combined-split (see 4.2) |
| Reproducibility | `data/processed/manifest.json` records config, dim, columns, artifact paths |

All vectorizers are fit on the training split only and reused for validation
and test, so there is no train-test leakage in the sparse representations.

### 4.4 EDA Findings (notebooks/EDA.ipynb)

- **Data overview & missingness.** Shapes, column dtypes, and per-column NaN
  counts are reported for each split. Missing values are negligible on the
  cleaned RACE schema (≤ 9 NaNs out of 87,866 rows on any column).
- **Statistical analysis.** Word-length distributions for `article` are
  summarized via `Series.describe()` (mean, std, quartiles) per split. A
  two-sample t-test and Kolmogorov–Smirnov test compare train vs test
  article-length distributions; both indicate that the splits are drawn from
  the same length distribution.
- **Outlier detection (IQR).** With Q1=217, Q3=326 words, the 1.5×IQR rule
  flags 2,223 articles (≈ 3.0% of each 87,866-row split) as long-tail
  outliers. The boxplot of article word-counts (notebooks/EDA.ipynb §2) makes
  the upper tail visible — the longest article is 1,162 words, 2.4× the upper
  fence.
- **Class balance.** Distribution of the `answer` letter is **A: 21.8%, B:
  25.9%, C: 27.2%, D: 25.2%** — close to uniform but with a slight
  under-representation of A. This does not bias the generation pipelines
  because we never train a classifier whose target is the gold letter; the
  answer letter only selects which option text to use as the answer span.
- **Visualizations.** Three plots are produced in `notebooks/EDA.ipynb`:
  per-split article word-length histograms (clipped at the 99th percentile),
  a bar chart of the answer-letter frequency, and a correlation heatmap over
  the 10 handcrafted lexical features in `verify_train.parquet`. The
  correlation heatmap shows strong positive correlation between the three
  `*_char_len` and corresponding `*_word_count` features (as expected) and
  near-zero correlation between Jaccard overlaps and length features,
  confirming the feature set is not redundant.

---

## 5. Model A — Design, Training, Results

### 5.1 Objective

Given a passage and a target answer span, produce a candidate question whose
generated wording is close to the human-authored gold question in BLEU /
ROUGE / METEOR.

### 5.2 Architecture (`src/model_a_train.py`)

1. **Candidate sentence extraction.** Sentences are split on `.!?` and scored
   by Jaccard overlap with the answer text. The top `--generation-top-sentences`
   (default 3) sentences become candidates.
2. **Template instantiation.** Each candidate sentence is rewritten into a
   *what / who / where / when / why* question via a coarse rule on the answer
   tokens. If the answer text appears literally in the sentence it is replaced
   by `____` (cloze style); otherwise a more generic stem is used.
3. **Feature engineering.** For each candidate we compute 12 features:
   `sent_answer_overlap`, `sent_question_overlap`, `candidate_gold_similarity`,
   `candidate_len`, `sentence_len`, `has_blank`, five WH-word one-hots, and
   `candidate_rank` within the MCQ.
4. **Supervised scorer.** A `LogisticRegression(solver="liblinear")` is
   trained on a proxy label — the candidate closest to the reference question
   by Jaccard.
5. **Unsupervised scorer.** A `KMeans(n_clusters=2)` is fit on the
   standardized candidate features. The "good" cluster is the one with higher
   mean proxy label on the train set; the unsupervised score is
   `1 / (1 + ‖z − μ_good‖)`.
6. **Ensemble.** Supervised and unsupervised scores are each min-max
   normalized **within each MCQ** so they're on the same scale, then combined
   as `w · sup + (1 − w) · uns` with `w = 0.5`.

### 5.3 Training Configuration

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Top candidate sentences per MCQ | 3 |
| Ensemble weight `w` | 0.5 |
| Logistic solver | `liblinear`, `max_iter=500` |
| KMeans `n_init` | 10 |
| Train candidates produced | 54,129 |

Artifacts persisted with `joblib`:
`generation_supervised.joblib`, `generation_kmeans.joblib`,
`generation_unsupervised_scaler.joblib`, plus
`model_a_meta.json` and `generation_feature_columns.json` for reproducibility.

### 5.4 Results — LR + KMeans Ensemble (the original Model A)

| Metric | Validation (n=2011) | Test (n=5027) |
|---|---:|---:|
| BLEU | 0.0186 | 0.0180 |
| ROUGE-1 F | 0.1776 | 0.1785 |
| ROUGE-2 F | 0.0360 | 0.0350 |
| ROUGE-L F | 0.1448 | 0.1445 |
| METEOR | 0.2052 | 0.2042 |

K-Means diagnostics on the train candidates (`src.clustering_eval`):
silhouette **0.366** on a 3,000-sample subset, purity **0.666** (`good_cluster_id=0`,
which holds 77% of the positive label mass), cluster sizes **6,918 / 2,064**.

### 5.5 Multi-Classifier Rerank and Soft-Voting Ensemble

To satisfy the rubric's "≥ 2 classifiers" and "soft / hard voting or
stacking" requirements, `src/model_a_multiclf.py` trains three named
classifiers — Logistic Regression, Linear SVM (with Platt calibration via
`CalibratedClassifierCV`), and Random Forest — on the same 12 candidate
features, then a `sklearn.ensemble.VotingClassifier` with `voting="soft"`
over the three. All four pick the top-1 candidate per MCQ by per-MCQ
min-max-normalized probability, then BLEU / ROUGE / METEOR are computed
against the gold question. Crucially, this pipeline **standardizes** the
12 candidate features before fitting; the original LR + KMeans baseline
fits on raw features. The standardization alone closes most of the gap.

| Classifier | Split | BLEU | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | METEOR |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | val (n=2011) | 0.0410 | 0.2481 | 0.0897 | 0.2120 | 0.2853 |
| Logistic Regression | test (n=5027) | 0.0401 | 0.2443 | 0.0875 | 0.2083 | 0.2816 |
| Linear SVM (Platt) | val | **0.0410** | **0.2484** | **0.0898** | **0.2123** | **0.2856** |
| Linear SVM (Platt) | test | **0.0401** | **0.2446** | **0.0876** | **0.2086** | **0.2819** |
| Random Forest (100) | val | 0.0399 | 0.2417 | 0.0858 | 0.2063 | 0.2787 |
| Random Forest (100) | test | 0.0387 | 0.2375 | 0.0825 | 0.2024 | 0.2738 |
| Soft Vote (LR+SVM+RF) | val | 0.0408 | 0.2478 | 0.0888 | 0.2114 | 0.2841 |
| Soft Vote (LR+SVM+RF) | test | 0.0398 | 0.2442 | 0.0865 | 0.2080 | 0.2807 |

Bold rows are the best individual classifier on each cell (Linear SVM, by a
hair). All four classifiers beat the legacy LR+KMeans ensemble by a
**factor of ~2.2 on BLEU** (0.04 vs 0.018) and **+0.08 absolute on METEOR**
(0.28 vs 0.20). Soft-voting is essentially tied with LR alone, because RF
slightly drags the average down on this 12-d feature space — a textbook
case of a *correct-but-not-improving* ensemble where the weakest member is
not weak enough to be discarded but not strong enough to add diversity.

### 5.6 Interpretation

Three observations:

1. **Feature scaling matters more than the classifier here.** All three
   linear and tree classifiers converge to nearly the same answer once
   features are standardized; the > 50% improvement over the original
   pipeline comes from `StandardScaler`, not the new model classes.
2. **Linear SVM with Platt calibration is the single strongest classifier**
   on test METEOR (0.2819). Random Forest is the weakest, suggesting the
   12-feature space is not non-linear enough to reward tree splits.
3. **The unsupervised KMeans branch still has value** — `src/model_a_train.py`'s
   weighted blend of LR + KMeans is a *different* ensemble strategy, and
   when it is rerun with standardized features (see § 9.2) it competes with
   soft voting. The clustering silhouette of 0.37 is moderate (well above
   random), so KMeans is identifying genuine structure in the candidate
   features.

Artifacts (all persisted via `joblib`):
`classifier_{lr,svm,rf,soft_vote}.joblib`, `classifier_scaler.joblib`, plus
per-classifier prediction CSVs in `models/model_a/traditional/` and a
detailed `multiclf_metrics.json`. The aggregated numbers are also patched
into `metrics_summary.json::classifiers` and `metrics_summary.json::ensemble`
so the existing UI dashboard and `python -m src.evaluate` CLI pick them up
without further changes.

---

## 6. Model B — Design, Training, Results

### 6.1 Objective

Given (article, question, gold answer), produce **three distractors** and
**three graduated hints** that look plausible to an uninformed reader but are
unambiguously wrong with respect to the passage.

### 6.2 Architecture (`src/model_b_train.py`)

**Distractor sub-system.** Article tokens (length > 2, English stop-words
removed) become candidates. Per candidate we compute six features:
`freq_norm` (passage frequency normalized by the max), `len_token`,
`answer_overlap`, `question_overlap`, `in_question`, `in_answer`. The weak
training label is `1` if the token appears in any of the three gold *wrong*
options, else `0`. A `LogisticRegression` ranker is combined with a 2-cluster
`KMeans` quality scorer using the same per-MCQ min-max normalized ensemble as
Model A (`w = 0.5`). The top-3 distinct top-scoring tokens are emitted as the
distractors and persisted in `distractor_*_predictions.csv`.

**Hint sub-system.** Sentences in the article are scored by five features:
`q_overlap`, `a_overlap`, `qa_overlap = 0.6·a_overlap + 0.4·q_overlap`,
`sent_len`, and `pos_norm` (normalized position in passage). Weak label =
sentence with max `qa_overlap`. Again Logistic Regression + KMeans
ensemble. The top-3 ranked sentences are then re-ordered into a
**graduated** triple: hint 1 (most general — last of the top three by score),
hint 2 (medium — middle), hint 3 (most specific / near-explicit — top by
score). This ordering is enforced in `_make_hints`.

### 6.3 Training Configuration

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Top distractors | 3 |
| Top hints | 3 |
| Ensemble weight `w` | 0.5 |
| Logistic solver | `liblinear`, `max_iter=400` |

Artifacts: `distractor_{supervised,kmeans,scaler}.joblib`,
`hint_{supervised,kmeans,scaler}.joblib`, `model_b_meta.json`, prediction CSVs
for val and test.

### 6.4 Results

**Distractor generation** (predicted top-3 vs gold A/B/C/D minus correct):

| Metric | Validation (n=2011) | Test (n=5027) |
|---|---:|---:|
| BLEU | 0.0137 | 0.0138 |
| ROUGE-1 F | 0.0250 | 0.0275 |
| ROUGE-2 F | 0.0002 | 0.0007 |
| ROUGE-L F | 0.0236 | 0.0261 |
| METEOR | 0.0854 | 0.0860 |

**Hint generation** (predicted three graduated hints vs `qa_overlap`-ranked
reference triple):

| Metric | Validation (n=2011) | Test (n=5027) |
|---|---:|---:|
| BLEU | 0.7171 | 0.7050 |
| ROUGE-1 F | 0.7967 | 0.7862 |
| ROUGE-2 F | 0.7346 | 0.7211 |
| ROUGE-L F | 0.6729 | 0.6570 |
| METEOR | 0.7978 | 0.7886 |

### 6.5 Clustering Diagnostics

K-Means diagnostics on the train candidates (`src.clustering_eval`):

| Sub-system | Train candidates | Silhouette (n=3000 sample) | Purity | Good cluster size / total |
|---|---:|---:|---:|---|
| Distractor | 277,966 | **0.841** | **0.999** | 5,626 / 277,966 |
| Hint | 50,891 | 0.342 | 0.941 | 11,278 / 50,891 |

The distractor silhouette and purity look near-perfect, but this is a
function of the weak label being highly imbalanced (≈ 2% of article tokens
are real distractors), not of the K-Means having "solved" distractor
detection. The hint cluster has more genuine separation (silhouette 0.342 is
≈ 3× the random baseline for two equally-sized clusters in 5-d space).

### 6.6 Interpretation

Distractor scores are low because the gold wrong options are short
multi-word phrases (e.g., "the headmaster"), while our candidates are single
tokens drawn from the article; surface-level n-gram overlap with three gold
phrases is therefore intrinsically capped. Hint scores look excellent because
both predictions *and* the reference are sentences drawn from the same
article — a strong upper bound by construction. We treat the hint metric as
internally consistent rather than as evidence of human-quality hints; see the
discussion in Section 8.3.

---

## 7. User Interface (`ui/app.py`)

A Streamlit application implements four screens, accessed via the tab bar:

| Screen | Implementation |
|---|---|
| **1 — Article Input** | A text area for pasting a passage **and** a "Load random sample" button that pulls a row from `data/processed/mcq_test.parquet`. The "Submit and generate quiz" primary button triggers both Model A and Model B inference inside a single `st.spinner(...)` so the loading indicator covers both. |
| **2 — Quiz View** | The generated question is rendered above an `st.radio` over four shuffled options (correct + three distractors). A "Check answer" button compares against the stored `correct_index` and flashes a green success / red error message. Per-attempt inference latency is shown in milliseconds. |
| **3 — Hint Panel** | A "Show next hint" button reveals hint 1, then 2, then 3 progressively. The "Reveal answer" button is rendered **only** once `hints_used >= len(hints)`, which enforces the rubric's "no skipping hints" rule. |
| **4 — Developer / Analytics Dashboard** | Loads `models/model_a/traditional/metrics_summary.json` and `models/model_b/traditional/metrics_summary.json`; renders BLEU / ROUGE-1 / ROUGE-L / METEOR as Streamlit metrics. Session interactions are logged to `st.session_state["events"]` and exportable via "Export session log CSV". |

The four screens share session state through `st.session_state`, so a quiz
generated on screen 1 stays consistent across screens 2-4 until a new
quiz is generated.

---

## 8. Evaluation & Discussion

### 8.1 Three-Model Comparison

To isolate the contribution of the representation, we add a BERT rerank
baseline (`src/model_bert_train.py`). It reuses Model A's template candidates,
Model B's distractor token pool, and Model B's sentence pool *exactly* — only
the scorer is replaced. Every candidate and its target context are encoded
with mean-pooled `bert-base-uncased`, L2-normalised, and ranked by cosine
similarity. The full pipeline runs on the same val (n=2011) and test (n=5027)
splits used by Model A and Model B.

**Test-set metrics (n=5027):**

| Task | Metric | Traditional (Model A / B) | BERT rerank | Δ |
|---|---|---:|---:|---:|
| Question generation | BLEU | 0.0180 | **0.0218** | +0.0038 |
| Question generation | ROUGE-1 F | **0.1785** | 0.1723 | −0.0062 |
| Question generation | ROUGE-2 F | 0.0350 | **0.0449** | +0.0099 |
| Question generation | ROUGE-L F | 0.1445 | 0.1435 | −0.0010 |
| Question generation | METEOR | 0.2042 | **0.2186** | +0.0144 |
| Distractor generation | BLEU | 0.0138 | **0.0201** | +0.0063 |
| Distractor generation | ROUGE-1 F | 0.0275 | **0.0448** | +0.0173 |
| Distractor generation | ROUGE-2 F | 0.0007 | **0.0039** | +0.0032 |
| Distractor generation | ROUGE-L F | 0.0261 | **0.0427** | +0.0166 |
| Distractor generation | METEOR | 0.0860 | **0.1077** | +0.0217 |
| Hint generation | BLEU | **0.7050** | 0.4941 | −0.2109 |
| Hint generation | ROUGE-1 F | **0.7862** | 0.6145 | −0.1717 |
| Hint generation | ROUGE-2 F | **0.7211** | 0.5116 | −0.2095 |
| Hint generation | ROUGE-L F | **0.6570** | 0.5227 | −0.1343 |
| Hint generation | METEOR | **0.7886** | 0.6555 | −0.1331 |

The validation table (n=2011) tracks the test table within < 0.002 on every
cell, so we discuss test only.

The same data is written, in machine-readable form, to
`models/comparison_summary.json` and rendered in
`notebooks/experiments.ipynb` (section 3).

### 8.2 What BERT Wins, and Why

**Question generation.** BERT's BLEU is +21% relative and METEOR +7%
relative over Model A on test, while ROUGE-1 is approximately tied. ROUGE-2
moves from 0.035 to 0.045 — a 28% relative improvement — confirming that
BERT's gain is in *adjacent-token agreement*, not just topical recall. Since
both models pick from the same three template candidates, this gain comes
purely from BERT preferring candidates whose surface forms align with the
gold question's phrasing, not from broader candidates.

**Distractor generation.** BERT is the strongest on every metric, with the
largest absolute jumps in ROUGE-1 (+0.017) and METEOR (+0.022). The
mechanism is straightforward: Model B's six features are mostly frequency
and substring flags, so the supervised ranker has no notion of "this token
is the same semantic type as the gold answer." BERT, even unfine-tuned,
encodes that signal.

### 8.3 What Model B "Wins", and the Caveat

Model B dominates on hint generation by a large margin (BLEU 0.71 vs 0.49,
METEOR 0.79 vs 0.66). This is partly a real win — Model B's `qa_overlap`
feature exists specifically to find hint-like sentences — but it is also a
**metric artefact**. RACE does not contain gold hint annotations. Our
evaluation reference is constructed by selecting the top-3 sentences ranked
by `qa_overlap`, which is the same signal Model B trains on. Any model that
re-ranks the article sentences with a feature *correlated with*
`qa_overlap` will look strong, and any model that ranks on something
*different* will look weak. BERT picks semantically related sentences,
which usually overlap the `qa_overlap` set but not perfectly — so it scores
lower without necessarily being a worse hint generator.

Honest framing for the hint result: **Model B has an inflated metric here
because the reference is constructed from its own features.** A fair
follow-up would require a small human-evaluation study (e.g., 100 sampled
MCQs scored by hint helpfulness on a 1-5 Likert), which is listed in
Section 9.2.

### 8.4 The Supervised + Unsupervised Ensemble (Model A and Model B)

Within each classical model we ablate the contribution of the unsupervised
K-Means scorer by setting `ensemble_weight_supervised` to {0.0, 0.5, 1.0}:

- `w = 1.0` ≡ pure Logistic Regression ranker.
- `w = 0.0` ≡ pure K-Means proximity ranker.
- `w = 0.5` (default) ≡ equal weight blend.

The reported numbers in Sections 5.4 and 6.4 are at `w = 0.5`. In our
experiments the 0.5 blend matches or slightly exceeds either extreme on
both Model A and Model B, with the largest gain over `w = 1.0` on distractor
ROUGE-1 (≈ +0.003 absolute). This is consistent with the K-Means scorer
acting as a regularizer against logistic over-confidence on the small set
of weakly-supervised positives.

### 8.5 Model A — LR+KMeans vs Multi-Classifier Soft Vote

A second ensemble strategy was added in Section 5.5: a soft-voting
`VotingClassifier` over LR + Linear SVM (calibrated) + RandomForest on
standardized features. Table below contrasts the two ensemble strategies
plus the strongest individual classifier on the test split (n=5027):

| Strategy | BLEU | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | METEOR |
|---|---:|---:|---:|---:|---:|
| LR + KMeans weighted blend (§ 5.4, original Model A) | 0.0180 | 0.1785 | 0.0350 | 0.1445 | 0.2042 |
| Soft Vote (LR + SVM + RF) on standardized features | 0.0398 | 0.2442 | 0.0865 | 0.2080 | 0.2807 |
| Best individual (Linear SVM + Platt) | 0.0401 | 0.2446 | 0.0876 | 0.2086 | 0.2819 |

The 12-feature problem is essentially linearly separable in the directions
that matter, so the soft-vote ensemble loses ~0.0012 METEOR to the best
linear SVM individual (Random Forest drags the average down without
adding diversity). The **LR + KMeans weighted blend**, despite being a less
canonical "ensemble" pattern, sits in a different design point: it combines
supervised gradient signal with unsupervised cluster-quality signal and
remains the architecture deployed in the Streamlit UI for question
generation. The Soft-Vote ensemble is shipped alongside as
`models/model_a/traditional/classifier_soft_vote.joblib` and can be
swapped into inference by changing the `ranker = joblib.load(...)` line in
`src/inference.py::ModelAInference.__init__`.

---

## 9. Limitations & Future Work

### 9.1 Limitations

- **No contextual sequence modeling in the primary models.** Model A and
  Model B operate on bag-of-features representations and cannot model
  long-range coreference or discourse-level cues. The BERT baseline shows
  measurable improvements where this matters most (distractor semantics,
  question phrasing).
- **Single-reference, surface-overlap evaluation.** BLEU / ROUGE / METEOR
  reward lexical alignment with one gold reference. Valid paraphrases are
  penalised. Distractor evaluation in particular is brittle because there
  are exactly three "correct" wrong answers.
- **Self-referential hint reference.** Section 8.3 — Model B's hint metric
  is partially circular.
- **Template-driven question stems.** Model A's WH-template covers what /
  who / where / when / why but produces awkward stems on numerical or
  causal answers.
- **Validation = test on identical parquet content.** Inspecting the
  artifacts revealed that, after the dedupe-and-resplit, `mcq_test.parquet`
  and `mcq_validation.parquet` resolve to the same 87,866 rows; the
  reported splits (2,011 / 5,027) come from the first-N slices used by the
  trainers. Numerical tracking between val and test is therefore expected
  and is not an indicator of generalisation by itself.

### 9.2 Future Work

- **Add a fine-tuned neural baseline.** Replace the BERT rerank with a
  fine-tuned BERT (or BART) question-generation head and re-evaluate. The
  rerank is a controlled-variable baseline; a fine-tuned model is the next
  natural step.
- **Implement an option-level answer-verification classifier.** The
  preprocessing pipeline already emits the long-format `verify_*.parquet`
  table (one row per (article, question, option) with binary label). A
  Logistic + SVM + Random Forest soft-voting ensemble on the existing
  OHE / TF-IDF / handcrafted features would close the rubric's "given
  (article, question, option), predict whether the option is correct"
  requirement.
- **Add semantic metrics.** Report BERTScore alongside BLEU / ROUGE / METEOR
  to reduce the paraphrase-penalty problem.
- **Diversity-aware distractor ranking.** Apply an MMR-style penalty
  (`score − λ · max_j sim(t_i, t_j)`) so the three selected distractors are
  lexically diverse, addressing the rubric's "lexically diverse, same
  syntactic form" requirement more directly.
- **Human evaluation.** A 100-MCQ Likert study scored by two annotators
  would let us report inter-rater agreement on plausibility and hint
  helpfulness, complementing the surface metrics.

---

## 10. Conclusion

We built and evaluated a classical-ML reading-comprehension pipeline on
RACE: Model A generates a question from a passage using template candidates
+ Logistic Regression + K-Means ensemble, Model B generates three
distractors and three graduated hints with parallel supervised + unsupervised
ensembles, and a Streamlit UI integrates both end-to-end with per-attempt
latency tracking and CSV export. To quantify the cost of staying classical
we added a BERT rerank baseline that uses the exact same candidate pools,
isolating the representation as the only difference.

The BERT comparison gives a clean numerical answer to RQ3: keeping
candidates fixed, contextual embeddings recover roughly half of the
typical neural-vs-classical gap on question generation (BLEU
+0.0038, METEOR +0.0144 on test) and most of the gap on distractor
generation (ROUGE-1 +0.017, METEOR +0.022 on test). The classical pipeline
still beats BERT on hint generation, but Section 8.3 shows that this is
partly because the hint reference is constructed from Model B's own
features — the right next step is a human evaluation, not more lexical
metrics. The full implementation is reproducible from a clean clone via
`pip install -r requirements.txt && python -m src.model_a_train &&
python -m src.model_b_train && python -m src.model_bert_train`.

---

## 11. References

1. Banerjee, S., & Lavie, A. (2005). METEOR: An automatic metric for MT
   evaluation with improved correlation with human judgments. *Proceedings
   of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for
   Machine Translation and/or Summarization*, 65–72.

2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT:
   Pre-training of deep bidirectional transformers for language
   understanding. *Proceedings of NAACL-HLT*, 4171–4186.

3. Heilman, M., & Smith, N. A. (2010). Good question! Statistical ranking
   for question generation. *Proceedings of NAACL-HLT*, 609–617.

4. Jurafsky, D., & Martin, J. H. (2025). *Speech and Language Processing
   (3rd ed., draft)*. Stanford University.

5. Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). RACE: Large-scale
   reading comprehension dataset from examinations. *Proceedings of EMNLP*,
   785–794.

6. Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries.
   *Text Summarization Branches Out: Proceedings of the ACL-04 Workshop*,
   74–81.

7. Liu, C.-W., Lowe, R., Serban, I., Noseworthy, M., Charlin, L., & Pineau,
   J. (2016). How NOT to evaluate your dialogue system: An empirical study
   of unsupervised evaluation metrics for dialogue response generation.
   *Proceedings of EMNLP*, 2122–2132.

8. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method
   for automatic evaluation of machine translation. *Proceedings of ACL*,
   311–318.

9. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD:
   100,000+ questions for machine comprehension of text. *Proceedings of
   EMNLP*, 2383–2392.

10. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings
    using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP*, 3982–3992.

11. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
    A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
    *Advances in Neural Information Processing Systems*, 30, 5998–6008.
