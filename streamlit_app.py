"""
Streamlit web UI for the Wound Care Literature Search Agent.

Run:
  streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import json
import logging
import os
from datetime import date, datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)


# ── Startup diagnostics ───────────────────────────────────────────────────────
# Catches silent import failures that cause a blank screen instead of an error.

_import_errors: list[str] = []

try:
    import pandas as _pd  # noqa: F401
except ImportError:
    _import_errors.append("pandas — run: pip install pandas")

try:
    from dotenv import load_dotenv as _ld  # noqa: F401
except ImportError:
    _import_errors.append("python-dotenv — run: pip install python-dotenv")

try:
    import requests as _req  # noqa: F401
except ImportError:
    _import_errors.append("requests — run: pip install requests")

try:
    import langchain  # noqa: F401
except ImportError:
    _import_errors.append("langchain — run: pip install langchain")

try:
    import langgraph  # noqa: F401
except ImportError:
    _import_errors.append("langgraph — run: pip install langgraph")

if _import_errors:
    st.error("**Missing packages — install them and restart Streamlit:**")
    for _e in _import_errors:
        st.code(f"pip install {_e.split(' — ')[0]}", language="bash")
    st.stop()

st.set_page_config(
    page_title="Wound Care Literature Search",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state initialisation ──────────────────────────────────────────────
# Results are stored here so that clicking download buttons or changing the
# article dropdown does NOT wipe the screen — data persists until a new search.

for _key, _default in {
    "search_results":  None,   # full result dict from agent.search()
    "search_query":    "",     # query string that produced the results
    "search_running":  False,  # guard against double-submit
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ── Download helpers ───────────────────────────────────────────────────────────

def _format_single_abstract(r) -> str:
    lines = [
        "=" * 70,
        f"PMID:     {r.pmid}",
        f"Title:    {r.title}",
        f"Authors:  {r.format_authors(10)}",
        f"Journal:  {r.journal}",
        f"Year:     {r.pub_date[:4] if r.pub_date else '—'}",
        f"Type:     {', '.join(r.article_type) if r.article_type else 'Article'}",
        f"DOI:      {r.doi or '—'}",
        f"URL:      {r.pubmed_url}",
    ]
    if r.mesh_terms:
        lines.append(f"MeSH:     {'; '.join(r.mesh_terms[:10])}")
    lines += ["", "ABSTRACT:", r.abstract or "No abstract available.", ""]
    return "\n".join(lines)


def _format_all_abstracts(articles, query: str = "") -> str:
    header = [
        "WOUND CARE LITERATURE SEARCH — ARTICLE ABSTRACTS",
        f"Query:     {query}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Articles:  {len(articles)}",
        "",
    ]
    blocks = ["\n".join(header)]
    for i, r in enumerate(articles, 1):
        blocks.append(f"[{i} of {len(articles)}]\n{_format_single_abstract(r)}")
    return "\n".join(blocks)


def _build_full_report(summary: str, articles, query: str = "") -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "=" * 70,
        "WOUND CARE LITERATURE SEARCH — FULL REPORT",
        f"Query:     {query}",
        f"Generated: {ts}",
        f"Articles:  {len(articles)}",
        "=" * 70,
        "",
        "SECTION 1 — CLINICAL EVIDENCE SUMMARY",
        "-" * 70,
        "",
        summary,
        "",
        "=" * 70,
        "SECTION 2 — ARTICLE ABSTRACTS",
        "-" * 70,
        "",
        _format_all_abstracts(articles, query),
    ]
    return "\n".join(parts)


# ── Agent loader (cached per provider+model so it survives reruns) ─────────────

@st.cache_resource(show_spinner=False)
def load_agent(provider: str, model: str, ncbi_key: str):
    from agent import WoundCareAgent
    from main import get_llm
    try:
        llm = get_llm(provider, model)
    except Exception as exc:
        return None, str(exc)
    return WoundCareAgent(llm, ncbi_api_key=ncbi_key), None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🩹 Wound Care Agent")
    st.markdown("---")

    st.subheader("LLM Provider")
    provider = st.selectbox("Provider", ["openai", "anthropic", "groq"])
    model_map = {
        "openai":    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "anthropic": ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "groq":      ["llama-3.3-70b-versatile", "Qwen/Qwen3-32B",
                      "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.1-8b-instant"],
    }
    model = st.selectbox("Model", model_map[provider])

    st.markdown("---")
    st.subheader("Search Parameters")
    max_results = st.slider("Max results", 5, 100, 20, 5)
    article_type = st.selectbox(
        "Article type filter",
        ["all", "review", "systematic_review", "meta-analysis",
         "rct", "clinical_trial", "guideline", "observational", "case_report"],
    )

    col1, col2 = st.columns(2)
    with col1:
        date_from_val = st.date_input("From date", value=None)
    with col2:
        date_to_val = st.date_input("To date", value=None)

    date_from = date_from_val.strftime("%Y/%m/%d") if date_from_val else ""
    date_to   = date_to_val.strftime("%Y/%m/%d")   if date_to_val   else ""

    st.markdown("---")
    st.subheader("API Keys")
    with st.expander("Enter API keys (or set in .env)"):
        llm_key  = st.text_input(f"{provider.upper()} Key", type="password",
                                  value=os.getenv(f"{provider.upper()}_API_KEY", ""))
        ncbi_key = st.text_input("NCBI API Key (optional)", type="password",
                                  value=os.getenv("NCBI_API_KEY", ""))
        if llm_key:
            os.environ[f"{provider.upper()}_API_KEY"] = llm_key
        if ncbi_key:
            os.environ["NCBI_API_KEY"] = ncbi_key

    st.markdown("---")
    # Clear results button in sidebar so a fresh search can always be started
    if st.session_state.search_results is not None:
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.search_results = None
            st.session_state.search_query   = ""
            st.rerun()

    st.caption("Searches PubMed via NCBI E-utilities. Respects rate limits.")


# ── Main area — search bar ────────────────────────────────────────────────────

st.title("Wound Care Literature Search")
st.markdown(
    "Search PubMed using natural language. MeSH terms are applied automatically "
    "for wound care topics."
)

query = st.text_input(
    "Search query",
    placeholder="e.g. negative pressure wound therapy for diabetic foot ulcers",
    help="Type a natural language wound care query. MeSH mapping is automatic.",
)

example_queries = [
    "negative pressure wound therapy diabetic foot ulcers",
    "best dressings for pressure injuries 2023-2026",
    "biofilm management chronic wounds",
]
st.caption("Examples: " + " · ".join(f"`{q}`" for q in example_queries))

search_btn = st.button("🔍 Search PubMed", type="primary", use_container_width=True)


# ── Run search — stores result in session_state, does NOT render yet ──────────

if search_btn and query.strip():
    agent, err = load_agent(provider, model, os.getenv("NCBI_API_KEY", ""))

    if err:
        st.error(f"Agent initialisation failed: {err}")
        st.info("Set your LLM API key in the sidebar or in a .env file.")
        st.stop()

    with st.spinner("Searching PubMed and summarising evidence…"):
        try:
            result = agent.search(
                query=query,
                max_results=max_results,
                article_type=article_type,
                date_from=date_from,
                date_to=date_to,
            )
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            st.stop()

    # Persist to session state — this survives all subsequent reruns
    st.session_state.search_results = result
    st.session_state.search_query   = query

elif search_btn and not query.strip():
    st.warning("Please enter a search query.")


# ── Render results from session_state ────────────────────────────────────────
# This block runs on EVERY rerun (button clicks, dropdown changes, etc.)
# because it reads from session_state, not from a one-time search trigger.

if st.session_state.search_results is not None:

    result       = st.session_state.search_results
    saved_query  = st.session_state.search_query
    articles     = result["articles"]
    meta         = result["search_metadata"]
    summary      = result.get("summary", "")

    # ── Stats bar
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total PubMed hits",  meta.get("total_count", "—"))
    c2.metric("Articles retrieved", len(articles))
    c3.metric("Study type filter",  article_type)
    c4.metric("Date range", f"{date_from or '∞'} → {date_to or '∞'}")

    with st.expander("PubMed query used"):
        st.code(meta.get("query_used", "N/A"), language="text")

    st.divider()

    if articles:

        # ── Tabs: Results Table | Evidence Summary ─────────────────────────────
        tab1, tab2 = st.tabs(["📋 Results Table", "📝 Evidence Summary"])

        # ── Tab 1 — Results table + CSV / JSON / BibTeX downloads ──────────────
        with tab1:
            rows = []
            for r in articles:
                rows.append({
                    "PMID":    r.pmid,
                    "Year":    r.pub_date[:4] if r.pub_date else "—",
                    "Title":   r.title,
                    "Authors": r.format_authors(2),
                    "Journal": r.journal,
                    "Type":    r.article_type[0] if r.article_type else "Article",
                    "DOI":     r.doi or "",
                    "URL":     r.pubmed_url,
                })

            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "URL":  st.column_config.LinkColumn("PubMed Link"),
                    "DOI":  st.column_config.LinkColumn("DOI"),
                    "PMID": st.column_config.TextColumn("PMID", width="small"),
                    "Year": st.column_config.TextColumn("Year", width="small"),
                },
                hide_index=True,
            )

            st.subheader("Export Results")
            dcol1, dcol2, dcol3 = st.columns(3)

            # CSV
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            dcol1.download_button(
                "⬇ Download CSV",
                data=csv_buf.getvalue(),
                file_name="wound_care_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # JSON
            json_data = json.dumps([r.to_dict() for r in articles], indent=2)
            dcol2.download_button(
                "⬇ Download JSON",
                data=json_data,
                file_name="wound_care_results.json",
                mime="application/json",
                use_container_width=True,
            )

            # BibTeX
            bibtex_lines: list[str] = []
            for r in articles:
                fa = r.authors[0].split()[-1] if r.authors else "Unknown"
                yr = r.pub_date[:4] if r.pub_date else "0000"
                bibtex_lines.append(
                    f"@article{{{fa}{yr}_{r.pmid},\n"
                    f"  title   = {{{{{r.title}}}}},\n"
                    f"  author  = {{{' and '.join(r.authors)}}},\n"
                    f"  journal = {{{r.journal}}},\n"
                    f"  year    = {{{yr}}},\n"
                    f"  url     = {{{r.pubmed_url}}}\n"
                    f"}}"
                )
            dcol3.download_button(
                "⬇ Download BibTeX",
                data="\n\n".join(bibtex_lines),
                file_name="wound_care_results.bib",
                mime="text/plain",
                use_container_width=True,
            )

        # ── Tab 2 — Evidence summary + downloads ───────────────────────────────
        with tab2:
            if summary:
                st.markdown(summary)
                st.divider()
                st.subheader("Download Evidence Summary")
                scol1, scol2 = st.columns(2)
                scol1.download_button(
                    "⬇ Download as Plain Text",
                    data=summary,
                    file_name="wound_care_evidence_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                scol2.download_button(
                    "⬇ Download as Markdown",
                    data=summary,
                    file_name="wound_care_evidence_summary.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.info("No summary generated.")

        # ── Abstract viewer ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Article Abstracts")

        pmid_options = {f"{r.pmid} — {r.title[:65]}": r for r in articles}

        # key= keeps the selection stable across reruns
        selected_label = st.selectbox(
            "Select an article to read its abstract",
            list(pmid_options.keys()),
            key="selected_article",
        )

        if selected_label:
            sel = pmid_options[selected_label]
            st.markdown(f"**{sel.title}**")
            st.caption(
                f"{sel.format_authors(5)} · {sel.journal} · {sel.pub_date[:4]} · "
                f"[PubMed]({sel.pubmed_url})"
                + (f" · [DOI](https://doi.org/{sel.doi})" if sel.doi else "")
            )
            if sel.mesh_terms:
                st.caption("MeSH: " + ", ".join(sel.mesh_terms[:10]))
            st.markdown(sel.abstract or "_No abstract available._")

            st.download_button(
                f"⬇ Download this abstract (PMID {sel.pmid})",
                data=_format_single_abstract(sel),
                file_name=f"abstract_{sel.pmid}.txt",
                mime="text/plain",
            )

        # ── Bulk downloads — all abstracts & full report ───────────────────────
        st.divider()
        st.subheader("Download All Abstracts")

        all_abstracts_txt = _format_all_abstracts(articles, saved_query)
        full_report_txt   = _build_full_report(summary, articles, saved_query)

        bcol1, bcol2 = st.columns(2)
        bcol1.download_button(
            "⬇ All Abstracts (TXT)",
            data=all_abstracts_txt,
            file_name="wound_care_all_abstracts.txt",
            mime="text/plain",
            use_container_width=True,
            help="Every article's abstract in one plain-text file",
        )
        bcol2.download_button(
            "⬇ Full Report — Summary + All Abstracts",
            data=full_report_txt,
            file_name="wound_care_full_report.txt",
            mime="text/plain",
            use_container_width=True,
            help="Evidence summary (Section 1) followed by every abstract (Section 2)",
        )

    else:
        st.warning("No articles found. Try broadening the query or removing date filters.")
