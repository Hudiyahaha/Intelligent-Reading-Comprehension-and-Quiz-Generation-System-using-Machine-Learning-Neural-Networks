import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import PreprocessConfig, clean_text, run_preprocessing


def test_clean_text_basic():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_mcq_pipeline_roundtrip(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "article": "The cat sat on the mat.",
                "question": "Where did the cat sit?",
                "A": "On the mat",
                "B": "On the roof",
                "C": "In the tree",
                "D": "Under the bed",
                "answer": "A",
            }
        ]
    )
    df.to_csv(raw / "train.csv", index=False)
    df.to_csv(raw / "dev.csv", index=False)
    df.to_csv(raw / "test.csv", index=False)

    processed = tmp_path / "processed"
    run_preprocessing(
        raw,
        processed,
        PreprocessConfig(max_ohe_features=256, max_tfidf_features=256, max_aq_tfidf_features=128, min_df=1),
    )

    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["train"] == 4  # four verification rows

    v = pd.read_parquet(processed / "verify_train.parquet")
    assert set(v["label"].unique()) == {0, 1}
    assert int(v["label"].sum()) == 1
    assert "feat_cosine_article_question" in v.columns
    assert "feat_jaccard_question_article" in v.columns
    assert np.isfinite(v["feat_cosine_article_question"].astype(float)).all()
