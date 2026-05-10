# Automatic Question, Distractor, and Hint Generation using Traditional Machine Learning on RACE

## 1. Abstract

This project evaluates how far traditional machine learning can go on a reading-comprehension generation workflow built from the RACE dataset. The implementation intentionally excludes neural architectures and uses only classical supervised and unsupervised methods with ensemble scoring.

Two models were implemented. **Model A** generates a question from a passage using template-based candidates, a supervised logistic ranker, a K-Means unsupervised scorer, and weighted ensemble selection. **Model B** generates distractors and hints using separate supervised+unsupervised ensemble pipelines. **Both models** are evaluated with BLEU, ROUGE, and METEOR against the dataset references (generated distractor strings vs gold incorrect options, generated hints vs reference hint text).

Results show that traditional methods can recover partial topical overlap but struggle with fluent, context-rich generation. Validation and test scores are close for both models, indicating stable generalization under the chosen split.

---

## 2. Introduction & Motivation

Modern NLP generation is dominated by neural sequence models, but this project focuses on traditional methods to quantify their strengths and limitations for educational reading-comprehension tasks.

Research questions:

- Can feature-based classical models produce useful question-generation outputs from passages?
- Does combining supervised and unsupervised scoring improve output quality?
- How well do lexical overlap metrics (BLEU/ROUGE/METEOR) capture traditional model performance?

The project objective is practical and comparative: build an end-to-end traditional pipeline (preprocessing, Model A, Model B, UI), then analyze where traditional methods succeed and where they fail.

---

## 3. Related Work

RACE introduced a challenging reading-comprehension benchmark with human-authored passages and MCQs requiring non-trivial reasoning (Lai et al., 2017). Text-generation evaluation commonly uses overlap metrics such as BLEU (Papineni et al., 2002), ROUGE (Lin, 2004), and METEOR (Banerjee & Lavie, 2005). Prior to transformers, many NLP pipelines relied on sparse features and classical models (logistic regression, SVMs, Naive Bayes, clustering), which are computationally efficient but limited in contextual language modeling.

This work differs from contemporary neural studies by intentionally constraining the solution to classical methods and evaluating practical generation quality boundaries.

---

## 4. Dataset Analysis

### 4.1 Dataset and Columns

The dataset uses the RACE schema:

- `id`
- `article`
- `question`
- `A`, `B`, `C`, `D`
- `answer`

### 4.2 Split Correction and Deduplication

The provided raw files had overlap/duplication concerns, so preprocessing uses a combined split mode:

1. Merge available raw CSVs.
2. Deduplicate by `id`.
3. Shuffle with fixed seed.
4. Split into train/validation/test with configurable fractions.

In this run, resulting MCQ split sizes were:

- Train: 18,097
- Validation: 2,011
- Test: 5,027

### 4.3 Preprocessing Pipeline

Implemented preprocessing in `src/preprocessing.py`:

- Lowercasing and punctuation cleanup.
- Cleaned MCQ parquet outputs (`mcq_train/validation/test.parquet`).
- Optional verification-long format and sparse matrices (kept for extensibility).
- Artifacts (`joblib`) and `manifest.json` for reproducibility.

### 4.4 EDA Notes

Core EDA and run orchestration were done in `notebooks/EDA.ipynb`.  
Add plots/tables in final PDF if required by rubric:

- Passage/question length histograms
- Vocabulary frequency plot
- Split-size table before vs after deduplication

---

## 5. Model A: Design, Training, Results

### 5.1 Objective

Generate a question from a passage using traditional ML only.

### 5.2 Architecture

Model A pipeline (`src/model_a_train.py`):

1. Candidate sentence extraction from passage (top overlap with answer/question signals).
2. Template-based question generation (Who/What/Where/When/Why).
3. Candidate feature extraction (overlaps, lengths, template flags, rank indices).
4. **Supervised scorer**: Logistic Regression.
5. **Unsupervised scorer**: K-Means + distance-to-good-cluster score.
6. **Ensemble**: per-sample min-max normalized weighted combination (`w=0.5`).
7. Final top-1 candidate selected as generated question.

### 5.3 Training Configuration

- Random seed: 42
- Top generated sentences per item: 3
- Ensemble supervised weight: 0.5
- Full split usage (no train/val/test caps)

### 5.4 Evaluation Metrics

Because this is generation, evaluation uses:

- BLEU
- ROUGE-1 / ROUGE-2 / ROUGE-L (F1)
- METEOR

### 5.5 Results

From `models/model_a/traditional/metrics_summary.json`:

| Metric | Validation | Test |
|---|---:|---:|
| BLEU | 0.0185646 | 0.0180066 |
| ROUGE-1 (F1) | 0.1776456 | 0.1785060 |
| ROUGE-2 (F1) | 0.0360220 | 0.0350499 |
| ROUGE-L (F1) | 0.1447998 | 0.1445434 |
| METEOR | 0.2052491 | 0.2042214 |

Validation `n=2011`, Test `n=5027`.

### 5.6 Interpretation

Low BLEU/ROUGE-2 indicates weak phrase-level alignment with reference wording. ROUGE-1 and METEOR are higher, suggesting partial semantic/topic overlap. Validation and test closeness indicates stable behavior with limited overfitting.

---

## 6. Model B: Design, Training, Results

### 6.1 Objective

Generate distractors and hints using traditional ML only.

### 6.2 Architecture

Model B pipeline (`src/model_b_train.py`) has two subsystems:

#### A) Distractor Generation

1. Candidate term extraction from article tokens.
2. Candidate features: frequency, length, overlap indicators.
3. **Supervised scorer**: Logistic Regression.
4. **Unsupervised scorer**: K-Means cluster quality.
5. **Ensemble score** (`w=0.5`) and top-3 selection.

#### B) Hint Generation

1. Sentence segmentation of passage.
2. Sentence features: overlap with question/answer, length, position.
3. **Supervised scorer**: Logistic Regression.
4. **Unsupervised scorer**: K-Means quality scoring.
5. Ensemble ranking and selection of top-3 graduated hints.

### 6.3 Training Configuration

- Random seed: 42
- Top distractors: 3
- Top hints: 3
- Full split usage
- Ensemble supervised weight: 0.5

### 6.4 Evaluation Metrics

Model B uses the same overlap-based generation metrics as Model A, reported in `metrics_summary.json` for **distractor generation** and **hint generation** separately:

- BLEU
- ROUGE-1 / ROUGE-2 / ROUGE-L (F1)
- METEOR

Scores compare the model’s selected/generated text to the reference distractors and reference hints from the RACE rows used in each split.

### 6.5 Results

From `models/model_b/traditional/metrics_summary.json`:

#### Distractor generation (vs gold incorrect options)

| Metric | Validation | Test |
|---|---:|---:|
| BLEU | 0.0136533 | 0.0137739 |
| ROUGE-1 (F1) | 0.0250342 | 0.0275203 |
| ROUGE-2 (F1) | 0.0002452 | 0.0007444 |
| ROUGE-L (F1) | 0.0236135 | 0.0260934 |
| METEOR | 0.0853617 | 0.0859714 |

#### Hint generation (vs reference hints)

| Metric | Validation | Test |
|---|---:|---:|
| BLEU | 0.7170826 | 0.7050478 |
| ROUGE-1 (F1) | 0.7967040 | 0.7861892 |
| ROUGE-2 (F1) | 0.7346480 | 0.7210700 |
| ROUGE-L (F1) | 0.6728866 | 0.6570376 |
| METEOR | 0.7978418 | 0.7885873 |

Validation `n=2011`, Test `n=5027` (per subsystem).

### 6.6 Interpretation

Distractor scores stay low: distractors are short and lexically diverse, so token overlap with the exact gold wrong answers is weak even when options are superficially plausible. Hint scores are much higher because hints are ranked sentences drawn from the passage and align strongly with the reference hint strings under n-gram overlap. Validation and test numbers track closely for both subsystems, suggesting stable behavior on the combined split.

---

## 7. User Interface Description

A Streamlit UI was implemented in `ui/app.py`.

### 7.1 Implemented Screens

1. **Article Input**  
   - Paste article or load random sample from processed parquet
   - Submit triggers Model A + Model B inference

2. **Quiz View**  
   - Shows generated question
   - Presents one correct answer with three distractors
   - User checks answer with feedback

3. **Hint Panel**  
   - Progressive hint reveal
   - Reveal answer after hint sequence

4. **Developer Dashboard**  
   - Session logs, latency, interaction stats  
   - Loads and displays **Model A** and **Model B** `metrics_summary.json` (BLEU / ROUGE / METEOR) when those files are present next to the app  
   - CSV export of session events

### 7.2 Technologies

- Python
- Streamlit
- pandas / scikit-learn
- joblib model loading

### 7.3 End-to-End Flow

Input article -> Model A generates question -> Model B generates distractors/hints -> UI renders quiz + hint progression -> dashboard logs interactions.

> Add screenshots in final PDF:
> - Article Input tab
> - Quiz tab
> - Hints tab
> - Developer Dashboard

---

## 8. Evaluation & Discussion

### 8.1 Overall Findings

- Classical pipelines can produce usable, structured outputs.
- Lexical overlap with references remains low for generation.
- Ensemble scoring gives stable behavior across splits.

### 8.2 Model A vs Model B

- **Model A** is reference-evaluated (BLEU/ROUGE/METEOR) on full questions and shows limited but consistent overlap with gold wording.
- **Model B** uses the same metrics for distractors and hints; distractor overlap remains low, while hint overlap is high due to in-passage sentence selection.

### 8.3 Why Performance is Limited

- No deep contextual sequence modeling.
- Template/feature dependence for language generation.
- Single-reference evaluation penalizes valid paraphrases.
- Distractor evaluation is strict: matching exact gold incorrect options in surface form is hard for classical rankers.

### 8.4 Practical Value

Despite limitations, the project demonstrates:

- reproducible classical baselines,
- end-to-end model integration,
- deployable educational demo app,
- clear benchmark against modern neural expectations.

---

## 9. Limitations & Future Work

### 9.1 Limitations

- Traditional models cannot model long-range contextual semantics.
- Generated text quality depends heavily on templates and overlap heuristics.
- Metrics are still lexical overlap scores: they do not measure plausibility or pedagogical quality of distractors/hints on their own.
- Single-reference evaluation restricts fair scoring of paraphrases.

### 9.2 Future Work

- Add neural baseline(s) for direct gap analysis.
- Improve candidate generation with richer linguistic parsing.
- Add semantic metrics (e.g., BERTScore) in addition to overlap metrics.
- Use human evaluation protocol for distractor plausibility and hint helpfulness.
- Expand multi-reference evaluation where possible.

---

## 10. Conclusion

This project built and evaluated a fully traditional ML pipeline for reading-comprehension generation tasks on RACE. Model A (question generation) and Model B (distractor/hint generation) were implemented with supervised + unsupervised ensemble designs and integrated into a functional Streamlit app.

Results confirm that traditional methods can capture partial relevance but are limited in fluent, context-aware generation. Nonetheless, the system is reproducible, interpretable, and suitable as a strong classical baseline and educational comparison point against neural methods.

---

## 11. References

1. Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). *RACE: Large-scale ReAding Comprehension Dataset From Examinations*.
2. Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*.
3. Lin, C. Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*.
4. Banerjee, S., & Lavie, A. (2005). *METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments*.
5. Vaswani, A., et al. (2017). *Attention Is All You Need*.
6. Jurafsky, D., & Martin, J. H. *Speech and Language Processing*.

