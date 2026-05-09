"""Streamlit app wiring Model A + Model B end-to-end."""

from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import ModelAInference, ModelBInference


st.set_page_config(page_title="RC Quiz Generator", layout="wide")


@st.cache_resource
def load_models() -> tuple[ModelAInference, ModelBInference]:
    model_a = ModelAInference(Path("models/model_a/traditional"))
    model_b = ModelBInference(Path("models/model_b/traditional"))
    return model_a, model_b


@st.cache_data
def load_samples() -> pd.DataFrame:
    # Prefer test samples for demo; fallback to validation.
    for split in ("test", "validation", "train"):
        p = Path("data/processed") / f"mcq_{split}.parquet"
        if p.is_file():
            return pd.read_parquet(p)
    raise FileNotFoundError("No mcq_*.parquet found. Run preprocessing first.")


def _init_state() -> None:
    st.session_state.setdefault("article_text", "")
    st.session_state.setdefault("sample_id", "")
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("answer_text", "")
    st.session_state.setdefault("options", [])
    st.session_state.setdefault("correct_index", -1)
    st.session_state.setdefault("hints", [])
    st.session_state.setdefault("hints_used", 0)
    st.session_state.setdefault("last_latency_ms", 0.0)
    st.session_state.setdefault("events", [])


def _append_event(is_correct: bool | None) -> None:
    st.session_state["events"].append(
        {
            "sample_id": st.session_state.get("sample_id", ""),
            "question": st.session_state.get("question", ""),
            "answer_text": st.session_state.get("answer_text", ""),
            "selected_is_correct": is_correct,
            "hints_used": st.session_state.get("hints_used", 0),
            "latency_ms": st.session_state.get("last_latency_ms", 0.0),
            "num_distractors": max(0, len(st.session_state.get("options", [])) - 1),
        }
    )


def _build_quiz(article: str, sample_id: str) -> None:
    model_a, model_b = load_models()
    t0 = time.perf_counter()

    # Pick an answer anchor token/phrase from article using simple heuristic.
    tokens = [t for t in article.split() if len(t) > 3]
    answer_text = random.choice(tokens) if tokens else "answer"
    q_out = model_a.generate_question(article, answer_text)
    question = q_out["question"]

    b_out = model_b.generate(article, question, answer_text)
    distractors = [d for d in b_out.get("distractors", []) if d and d != answer_text]
    hints = [h for h in b_out.get("hints", []) if h]

    # Ensure 3 distractors; if model gives fewer, backfill with placeholders.
    fallback = ["option one", "option two", "option three", "option four"]
    for f in fallback:
        if len(distractors) >= 3:
            break
        if f != answer_text and f not in distractors:
            distractors.append(f)
    distractors = distractors[:3]

    options = [answer_text] + distractors
    random.shuffle(options)
    correct_index = options.index(answer_text)

    st.session_state["sample_id"] = sample_id
    st.session_state["question"] = question
    st.session_state["answer_text"] = answer_text
    st.session_state["options"] = options
    st.session_state["correct_index"] = correct_index
    st.session_state["hints"] = hints[:3]
    st.session_state["hints_used"] = 0
    st.session_state["last_latency_ms"] = (time.perf_counter() - t0) * 1000.0


def main() -> None:
    _init_state()
    st.title("Intelligent Reading Comprehension Quiz Generator")
    st.caption("Classical ML pipeline: Model A (question) + Model B (distractors/hints)")

    try:
        samples = load_samples()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load processed samples: {exc}")
        st.info("Run preprocessing first so data/processed/mcq_*.parquet exists.")
        return

    tabs = st.tabs(["Article Input", "Quiz", "Hints", "Developer Dashboard"])

    with tabs[0]:
        st.subheader("Screen 1: Article Input")
        c1, c2 = st.columns([2, 1])
        with c1:
            article = st.text_area(
                "Paste article",
                value=st.session_state["article_text"],
                height=220,
                placeholder="Paste a reading passage here...",
            )
        with c2:
            if st.button("Load random sample"):
                row = samples.sample(n=1, random_state=random.randint(1, 10_000)).iloc[0]
                st.session_state["article_text"] = str(row["article"])
                st.session_state["sample_id"] = str(row.get("id", "sample"))
                st.rerun()
            st.write("Current sample:", st.session_state.get("sample_id", "manual"))
        if st.button("Submit and generate quiz", type="primary"):
            if not article.strip():
                st.error("Please paste an article first.")
            else:
                st.session_state["article_text"] = article
                with st.spinner("Running Model A and Model B..."):
                    _build_quiz(article, st.session_state.get("sample_id", "manual"))
                st.success("Quiz generated. Open the Quiz tab.")

    with tabs[1]:
        st.subheader("Screen 2: Question & Answer Quiz")
        if not st.session_state["question"]:
            st.info("Generate a quiz from the Article Input tab first.")
        else:
            st.write(st.session_state["question"])
            opt = st.radio("Choose an option:", st.session_state["options"], index=0)
            if st.button("Check answer"):
                selected_idx = st.session_state["options"].index(opt)
                correct = selected_idx == st.session_state["correct_index"]
                if correct:
                    st.success("Correct! ✅")
                else:
                    st.error("Incorrect. ❌ Try hints in the Hints tab.")
                _append_event(correct)
            st.caption(f"Inference latency: {st.session_state['last_latency_ms']:.1f} ms")

    with tabs[2]:
        st.subheader("Screen 3: Hint Panel")
        hints = st.session_state.get("hints", [])
        if not hints:
            st.info("No hints yet. Generate a quiz first.")
        else:
            used = st.session_state["hints_used"]
            if st.button("Show next hint") and used < len(hints):
                st.session_state["hints_used"] = used + 1
                st.rerun()
            for i in range(st.session_state["hints_used"]):
                st.write(f"Hint {i + 1}: {hints[i]}")
            if st.session_state["hints_used"] >= len(hints):
                if st.button("Reveal answer"):
                    st.info(f"Answer: {st.session_state['answer_text']}")
                    _append_event(None)

    with tabs[3]:
        st.subheader("Screen 4: Developer / Analytics Dashboard")
        events = pd.DataFrame(st.session_state["events"])
        if events.empty:
            st.info("No interaction logs yet.")
        else:
            st.metric("Events logged", len(events))
            if events["selected_is_correct"].dropna().empty:
                st.metric("Quiz accuracy", "N/A")
            else:
                acc = float(events["selected_is_correct"].dropna().mean())
                st.metric("Quiz accuracy", f"{acc:.2%}")
            st.metric("Avg latency (ms)", f"{events['latency_ms'].mean():.1f}")
            st.dataframe(events, use_container_width=True)
            csv = events.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export session log CSV",
                data=csv,
                file_name="session_results.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
