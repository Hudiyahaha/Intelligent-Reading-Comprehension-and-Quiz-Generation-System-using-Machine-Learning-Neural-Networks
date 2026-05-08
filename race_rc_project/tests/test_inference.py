"""Smoke tests for ModelAInference and ModelBInference.

These tests load the joblib artifacts that ``src.model_a_train`` and
``src.model_b_train`` produce, run a single inference, and assert that the
returned payloads have the expected shape. If the artifacts are not present
(e.g. on a fresh clone before training) the tests are skipped instead of
failing, so the suite still passes during preprocessing-only runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest


MODEL_A_DIR = Path("models/model_a/traditional")
MODEL_B_DIR = Path("models/model_b/traditional")

_ARTICLE = (
    "Maria moved to Berlin in 2018 to study computer science. "
    "She lived in a small apartment near the university and worked part-time "
    "at a local cafe. On weekends she explored museums and parks across the city. "
    "After three years she finished her degree and joined a research lab focused "
    "on natural language processing."
)
_ANSWER_TEXT = "Berlin"
_QUESTION = "Where did Maria move in 2018?"


def _model_a_ready() -> bool:
    expected = [
        "generation_supervised.joblib",
        "generation_kmeans.joblib",
        "generation_unsupervised_scaler.joblib",
        "model_a_meta.json",
    ]
    return all((MODEL_A_DIR / f).is_file() for f in expected)


def _model_b_ready() -> bool:
    expected = [
        "distractor_supervised.joblib",
        "distractor_kmeans.joblib",
        "distractor_scaler.joblib",
        "hint_supervised.joblib",
        "hint_kmeans.joblib",
        "hint_scaler.joblib",
        "model_b_meta.json",
    ]
    return all((MODEL_B_DIR / f).is_file() for f in expected)


@pytest.mark.skipif(not _model_a_ready(), reason="Model A artifacts not built")
def test_model_a_generate_question_payload():
    from src.inference import ModelAInference

    model_a = ModelAInference(MODEL_A_DIR)
    out = model_a.generate_question(_ARTICLE, _ANSWER_TEXT)

    assert isinstance(out, dict)
    assert {"question", "ensemble_score"} <= set(out)
    assert isinstance(out["question"], str) and out["question"].strip()
    assert isinstance(out["ensemble_score"], float)
    assert 0.0 <= out["ensemble_score"] <= 1.0


@pytest.mark.skipif(not _model_a_ready(), reason="Model A artifacts not built")
def test_model_a_handles_empty_article_gracefully():
    from src.inference import ModelAInference

    model_a = ModelAInference(MODEL_A_DIR)
    out = model_a.generate_question("", _ANSWER_TEXT)
    assert isinstance(out, dict)
    assert "question" in out and isinstance(out["question"], str)


@pytest.mark.skipif(not _model_b_ready(), reason="Model B artifacts not built")
def test_model_b_generate_payload():
    from src.inference import ModelBInference

    model_b = ModelBInference(MODEL_B_DIR)
    out = model_b.generate(_ARTICLE, _QUESTION, _ANSWER_TEXT)

    assert isinstance(out, dict)
    assert {"distractors", "hints"} <= set(out)
    assert isinstance(out["distractors"], list)
    assert isinstance(out["hints"], list)
    for d in out["distractors"]:
        assert isinstance(d, str)
    for h in out["hints"]:
        assert isinstance(h, str)


@pytest.mark.skipif(not _model_b_ready(), reason="Model B artifacts not built")
def test_model_b_returns_top_k_distractors_and_hints():
    """The trained Model B is configured for top-3 distractors and top-3 hints."""
    from src.inference import ModelBInference

    model_b = ModelBInference(MODEL_B_DIR)
    out = model_b.generate(_ARTICLE, _QUESTION, _ANSWER_TEXT)
    assert len(out["distractors"]) <= model_b.top_d
    assert len(out["hints"]) <= model_b.top_h
    assert _ANSWER_TEXT not in out["distractors"]


@pytest.mark.skipif(
    not (_model_a_ready() and _model_b_ready()),
    reason="Both Model A and Model B artifacts required",
)
def test_end_to_end_quiz_assembly():
    """Model A produces a question, Model B produces distractors+hints, options assemble cleanly."""
    from src.inference import ModelAInference, ModelBInference

    model_a = ModelAInference(MODEL_A_DIR)
    model_b = ModelBInference(MODEL_B_DIR)

    q_out = model_a.generate_question(_ARTICLE, _ANSWER_TEXT)
    question = q_out["question"]
    assert question.strip()

    b_out = model_b.generate(_ARTICLE, question, _ANSWER_TEXT)
    distractors = b_out["distractors"]
    options = [_ANSWER_TEXT] + [d for d in distractors if d and d != _ANSWER_TEXT]
    assert len(set(options)) == len(options), "options must be unique"
    assert _ANSWER_TEXT in options
