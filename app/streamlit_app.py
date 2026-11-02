"""Streamlit front end. It holds no RAG logic: everything goes through the FastAPI backend.

Run:  streamlit run app/streamlit_app.py      (backend: uvicorn src.api.main:app)
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
PERSONAS = {
    "default": "Default",
    "formal": "Formal",
    "friendly": "Friendly",
    "concise": "Concise",
    "eli5": "Explain like I'm 5",
}

st.set_page_config(page_title="PDF RAG", page_icon="📄", layout="wide")


def api(method: str, path: str, timeout: int = 180, **kwargs):
    """One place for every backend call; returns (json_or_None, error_message_or_None)."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_URL} ({type(exc).__name__}). Is `uvicorn src.api.main:app` running?"
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"{response.status_code}: {detail}"
    return response.json(), None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📄 PDF RAG")
    health, err = api("GET", "/health", timeout=10)
    if err:
        st.error(err)
    else:
        st.caption(
            f"Index: **{health['chunks_indexed']}** chunks · "
            f"Gate: **{'trained ' + health['gate_version'] if health['gate_loaded'] else 'thresholds only'}**"
        )
        st.caption(
            f"Groq: {'✅' if health['groq_configured'] else '❌'} · "
            f"Azure backup: {'✅' if health['azure_configured'] else '— not configured'}"
        )

    st.divider()
    persona = st.selectbox("Answer style", list(PERSONAS), format_func=PERSONAS.get)
    explain = st.toggle("Explain the in-PDF decision (SHAP)", value=True)

    st.divider()
    st.subheader("Add a PDF")
    upload = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")
    with_images = st.checkbox("Describe figures with Gemini", value=False, help="Slower; uses Gemini quota.")
    if upload and st.button("Ingest", use_container_width=True):
        with st.spinner("Extracting, embedding and indexing..."):
            result, err = api("POST", f"/ingest?images={str(with_images).lower()}", timeout=900,
                              files={"file": (upload.name, upload.getvalue(), "application/pdf")})
        if err:
            st.error(err)
        else:
            st.success(f"{result['source']}: {result['pages']} pages, {result['chunks']} chunks in {result['seconds']:.0f}s")
            st.rerun()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()


# ---------------------------------------------------------------- helpers
def render_explanation(explanation: dict):
    base, prob = explanation["base_value"], explanation["probability"]
    st.markdown(
        f"The gate starts from an average question (**P = {base:.2f}**) and each signal moves it "
        f"to the final **P(in PDF) = {prob:.2f}**. Green pushes toward *in the PDF*, red toward *not in the PDF*."
    )
    frame = pd.DataFrame(explanation["contributions"])
    frame["label"] = frame["feature"] + " = " + frame["value"].map(lambda v: f"{v:.2f}")   # short names: long ones get cut off
    frame["toward the PDF"] = frame["shap"].clip(lower=0)
    frame["away from the PDF"] = frame["shap"].clip(upper=0)
    st.bar_chart(
        frame.set_index("label")[["toward the PDF", "away from the PDF"]],
        horizontal=True, color=["#2e9e5b", "#d9534f"],
    )
    st.caption("Bars are SHAP values in probability units. They add up exactly to (final − average).")
    st.dataframe(
        frame[["feature", "description", "value", "shap"]].rename(columns={"shap": "SHAP"}),
        use_container_width=True, hide_index=True,
    )


def render_answer(data: dict):
    if data["in_pdf"]:
        st.success("✔ Answered from the PDF")
    else:
        st.warning("⚠ NOT FROM THE PDF — answered from the model's general knowledge")
    st.markdown(data["answer"])

    cols = st.columns(4)
    cols[0].metric("Top relevance", f"{data['top_score']:.2f}")
    cols[1].metric("Gate P(in PDF)", "-" if data["gate_probability"] is None else f"{data['gate_probability']:.2f}")
    cols[2].metric("LLM", data["provider"] or "-")
    cols[3].metric("Latency", f"{data['timings_ms'].get('total_ms', 0) / 1000:.1f}s")

    if data["in_pdf"] and data["sources"]:
        with st.expander(f"Sources ({len(data['sources'])} passages)"):
            for i, src in enumerate(data["sources"], start=1):
                tag = " 🖼 from image" if src["kind"] == "image" else ""
                st.markdown(f"**[{i}] {src['source']} — page {src['page']}**{tag} · relevance {src['rerank_score']:.2f}")
                st.caption(src["text"][:500])
    with st.expander("How was this decided?"):
        st.write(data["verdict_reason"])
        st.code(" → ".join(data["path"]), language=None)
        for q in data["rewritten_queries"]:
            st.write(f"Retried search with: *{q}*")
        if data.get("explanation"):
            render_explanation(data["explanation"])


# ---------------------------------------------------------------- tabs
chat_tab, monitor_tab, about_tab = st.tabs(["💬 Chat", "📊 Monitoring", "ℹ️ About"])

with chat_tab:
    st.session_state.setdefault("messages", [])
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message["data"])

    if question := st.chat_input("Ask a question about your PDF"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching, checking scope, writing the answer..."):
                data, err = api("POST", "/chat", json={"question": question, "persona": persona, "explain": explain})
            if err:
                st.error(err)
            else:
                render_answer(data)
                st.session_state.messages.append({"role": "assistant", "data": data})

with monitor_tab:
    if st.button("Refresh"):
        st.rerun()
    stats, err = api("GET", "/stats", timeout=15)
    if err:
        st.error(err)
    elif stats["queries"] == 0:
        st.info("No questions logged yet.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Questions", stats["queries"])
        b.metric("Answered from PDF", stats["in_pdf"])
        c.metric("Out of PDF", stats["out_of_pdf"])
        d.metric("Latency avg / p95", f"{stats['latency_ms']['avg'] / 1000:.1f}s / {stats['latency_ms']['p95'] / 1000:.1f}s")

        left, right = st.columns(2)
        with left:
            st.subheader("What decided the verdict")
            st.bar_chart(pd.Series(stats["decided_by"], name="questions"))
        with right:
            st.subheader("Which LLM answered")
            st.bar_chart(pd.Series(stats["providers"], name="answers"))

        llm = stats.get("llm")
        if llm:
            st.caption(
                f"This server session: {llm['calls']} LLM calls · {llm['fallbacks']} served by the backup · "
                f"{llm['failures']} total failures · per provider {llm['by_provider']}"
            )
        st.subheader("Recent questions")
        st.dataframe(pd.DataFrame(stats["recent"]), use_container_width=True, hide_index=True)

    gate = (stats or {}).get("gate") if not err else None
    if gate:
        st.subheader(f"Answerability gate {gate['version']}")
        st.json({k: gate[k] for k in ("model", "n_rows", "test", "baseline_fixed_threshold", "cross_validation") if k in gate})

with about_tab:
    st.markdown(
        """
**How a question is answered**

1. **Retrieve**: dense (embedding) search + BM25 keyword search → fused with Reciprocal Rank Fusion → re-scored by a cross-encoder.
2. **Gate**: a trained classifier decides whether the PDF really answers the question (borderline cases ask the LLM).
3. **Answer**: in-PDF answers cite pages; out-of-PDF answers carry a visible warning.
4. **Fallback**: Groq writes the answer; if it fails, Azure OpenAI takes over automatically.
        """
    )
