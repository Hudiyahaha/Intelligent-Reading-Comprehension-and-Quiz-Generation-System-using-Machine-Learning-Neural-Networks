"""Unified inference API (Model A currently implemented)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.model_a_train import _guess_wh_word, _split_sentences, _template_question, _token_set
from src.preprocessing import add_handcrafted_lexical_features, clean_text


class ModelAInference:
    """Loads Model A artifacts and serves verification + question generation."""

    def __init__(
        self,
        processed_dir: Path | str = Path("data/processed"),
        model_dir: Path | str = Path("models/model_a/traditional"),
        verifier_name: str = "logistic_regression",
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.model_dir = Path(model_dir)
        manifest = json.loads((self.processed_dir / "manifest.json").read_text(encoding="utf-8"))
        self.handcrafted_cols: list[str] = manifest["handcrafted_feature_columns"]
        self.ohe = joblib.load(self.processed_dir / "artifacts" / "vectorizer_ohe.joblib")
        self.aq_tfidf = joblib.load(self.processed_dir / "artifacts" / "vectorizer_article_question_tfidf.joblib")
        self.verifier = joblib.load(self.model_dir / f"{verifier_name}.joblib")
        self.generation_ranker = joblib.load(self.model_dir / "generation_ranker.joblib")
        self.generation_feature_cols: list[str] = json.loads(
            (self.model_dir / "generation_feature_columns.json").read_text(encoding="utf-8")
        )

    def _verify_features(self, article: str, question: str, option: str):
        row = {
            "article_clean": clean_text(article),
            "question_clean": clean_text(question),
            "option_clean": clean_text(option),
        }
        df = pd.DataFrame([row])
        df = add_handcrafted_lexical_features(df)

        xa = self.aq_tfidf.transform([row["article_clean"]])
        xq = self.aq_tfidf.transform([row["question_clean"]])
        df["feat_cosine_article_question"] = cosine_similarity(xa, xq).diagonal().astype(np.float32)

        sparse_text = self.ohe.transform(
            [f"{row['article_clean']} {row['question_clean']} {row['option_clean']}"]
        )
        dense = df[self.handcrafted_cols].to_numpy(dtype=np.float32)
        return sparse_text, dense

    def verify_option(self, article: str, question: str, option: str) -> dict:
        x_sparse, dense = self._verify_features(article, question, option)
        # Most trained models expect sparse + handcrafted concatenated features.
        x = sparse.hstack([x_sparse, sparse.csr_matrix(dense)], format="csr")
        if hasattr(self.verifier, "predict_proba"):
            p = float(self.verifier.predict_proba(x)[0, 1])
            pred = int(p >= 0.5)
            return {"is_correct": pred == 1, "score": p}
        pred = int(self.verifier.predict(x)[0])
        return {"is_correct": pred == 1, "score": float(pred)}

    def generate_question(self, article: str, answer_text: str) -> dict:
        wh = _guess_wh_word(answer_text)
        sentences = _split_sentences(article)
        if not sentences:
            return {"question": "What is the correct answer?", "score": 0.0}

        records = []
        ans_tokens = _token_set(answer_text)
        for rank, s in enumerate(sentences[:10]):
            q, has_blank = _template_question(s, answer_text, wh)
            sent_tokens = _token_set(s)
            q_tokens = _token_set(q)
            overlap_ans = len(sent_tokens & ans_tokens) / max(1, len(sent_tokens | ans_tokens))
            overlap_q = len(sent_tokens & q_tokens) / max(1, len(sent_tokens | q_tokens))
            records.append(
                {
                    "candidate_question": q,
                    "sent_answer_overlap": overlap_ans,
                    "sent_question_overlap": overlap_q,
                    "candidate_gold_similarity": overlap_q,
                    "candidate_len": len(q_tokens),
                    "sentence_len": len(sent_tokens),
                    "has_blank": has_blank,
                    "wh_what": int(wh == "what"),
                    "wh_who": int(wh == "who"),
                    "wh_where": int(wh == "where"),
                    "wh_when": int(wh == "when"),
                    "wh_why": int(wh == "why"),
                    "candidate_rank": rank,
                }
            )
        cands = pd.DataFrame(records)
        X = cands[self.generation_feature_cols].to_numpy(dtype=np.float32)
        probs = self.generation_ranker.predict_proba(X)[:, 1]
        best_i = int(np.argmax(probs))
        return {
            "question": str(cands.iloc[best_i]["candidate_question"]),
            "score": float(probs[best_i]),
        }


def run_pipeline(*_args, **_kwargs):
    raise NotImplementedError("Model B pipeline is pending; use ModelAInference for now.")


if __name__ == "__main__":
    run_pipeline()
