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
import re
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

# ── Google Analytics 4 ────────────────────────────────────────────────────────
_ga_id = os.getenv("GA_MEASUREMENT_ID", "")
if _ga_id:
    st.html(f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={_ga_id}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{_ga_id}');
</script>
""")


# ── Session state initialisation ──────────────────────────────────────────────
# Results are stored here so that clicking download buttons or changing the
# article dropdown does NOT wipe the screen — data persists until a new search.

for _key, _default in {
    "search_results":   None,   # full result dict from agent.search()
    "cochrane_results": None,   # dict from search_cochrane()
    "search_query":     "",     # query string that produced the results
    "search_running":   False,  # guard against double-submit
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ── Google login gate ─────────────────────────────────────────────────────────
# Requires [auth] in .streamlit/secrets.toml (local) or equivalent secrets on
# the deployment platform — see .streamlit/secrets.toml.example for the fields.

if not st.user.is_logged_in:
    st.markdown("## 🔐 Wound Care Literature Search")
    st.markdown("Sign in with your Google account to continue.")
    st.button("Sign in with Google", type="primary", on_click=st.login)
    st.stop()


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


# ── PMID link helper ──────────────────────────────────────────────────────────
# Matches "PMID 12345678" optionally followed by ", 23456789, 34567890 ..."
# so that every number in a comma-separated group becomes a link while
# the "PMID" label itself stays as plain text.

_PMID_GROUP_RE = re.compile(r'\bPMID\s*:?\s*(\d+)((?:\s*,\s*\d+)*)')

def _linkify_pmids(text: str) -> str:
    """Turn PMID references into PubMed links (number only, new tab)."""
    def _link(pmid: str) -> str:
        return (f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/"'
                f' target="_blank">{pmid}</a>')

    def _replace(m: re.Match) -> str:
        result = f'PMID {_link(m.group(1))}'
        for extra in re.findall(r'\d+', m.group(2) or ""):
            result += f', {_link(extra)}'
        return result

    return _PMID_GROUP_RE.sub(_replace, text)


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
    st.caption(f"Signed in as {st.user.email}")
    if st.button("Log out", use_container_width=True):
        st.logout()
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
    max_results = st.slider("Max results", 5, 25, 5, 5)
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
    # Clear results button in sidebar so a fresh search can always be started
    if st.session_state.search_results is not None:
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.search_results   = None
            st.session_state.cochrane_results = None
            st.session_state.search_query     = ""
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
    "autologous blood clot therapy",
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

    with st.spinner("Searching PubMed and Cochrane, summarising evidence…"):
        try:
            result = agent.search(
                query=query,
                max_results=max_results,
                article_type=article_type,
                date_from=date_from,
                date_to=date_to,
            )
        except Exception as exc:
            st.error(f"PubMed search failed: {exc}")
            st.stop()

        try:
            from tools import search_cochrane
            cochrane = search_cochrane(
                query=query,
                max_results=max_results,
                date_from=date_from,
                date_to=date_to,
                api_key=os.getenv("NCBI_API_KEY", ""),
            )
        except Exception as exc:
            st.warning(f"Cochrane search failed: {exc}")
            cochrane = {"reviews": [], "central": [], "review_total": 0, "central_total": 0,
                        "review_query": "", "central_query": ""}

    # Persist to session state — survives all subsequent reruns
    st.session_state.search_results   = result
    st.session_state.cochrane_results = cochrane
    st.session_state.search_query     = query

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

        # ── Tabs: Evidence Summary | PubMed Results | Cochrane ────────────────
        tab1, tab2, tab3 = st.tabs(["📝 Evidence Summary", "📋 PubMed Results", "🔬 Cochrane"])

        # ── Tab 2 — Results table + CSV / JSON / BibTeX downloads ──────────────
        with tab2:
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

        # ── Tab 1 — Evidence summary + downloads ───────────────────────────────
        with tab1:
            if summary:
                st.markdown(_linkify_pmids(summary), unsafe_allow_html=True)
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

        # ── Tab 3 — Cochrane Reviews + CENTRAL ────────────────────────────────
        with tab3:
            coch = st.session_state.cochrane_results or {}
            c_reviews = coch.get("reviews", [])
            c_central = coch.get("central", [])

            cc1, cc2 = st.columns(2)
            cc1.metric("Cochrane Reviews (total hits)", coch.get("review_total", 0))
            cc2.metric("CENTRAL trials (total hits)",   coch.get("central_total", 0))

            # ── Cochrane Systematic Reviews ────────────────────────────────────
            st.subheader("Cochrane Systematic Reviews")
            if c_reviews:
                with st.expander("PubMed query used"):
                    st.code(coch.get("review_query", ""), language="text")

                rev_rows = [{
                    "PMID":    r.pmid,
                    "Year":    r.pub_date[:4] if r.pub_date else "—",
                    "Title":   r.title,
                    "Authors": r.format_authors(2),
                    "Journal": r.journal,
                    "DOI":     r.doi or "",
                    "URL":     r.pubmed_url,
                } for r in c_reviews]
                df_rev = pd.DataFrame(rev_rows)
                st.dataframe(df_rev, use_container_width=True,
                    column_config={
                        "URL": st.column_config.LinkColumn("PubMed Link"),
                        "DOI": st.column_config.LinkColumn("DOI"),
                        "PMID": st.column_config.TextColumn("PMID", width="small"),
                        "Year": st.column_config.TextColumn("Year", width="small"),
                    }, hide_index=True)

                rev_sel = st.selectbox("Select a Cochrane Review",
                    [f"{r.pmid} — {r.title[:65]}" for r in c_reviews],
                    key="selected_cochrane_review")
                if rev_sel:
                    sel_r = next(r for r in c_reviews if rev_sel.startswith(r.pmid))
                    st.markdown(f"**{sel_r.title}**")
                    st.caption(f"{sel_r.format_authors(5)} · {sel_r.journal} · "
                               f"{sel_r.pub_date[:4]}" +
                               (f" · [DOI](https://doi.org/{sel_r.doi})" if sel_r.doi else "") +
                               f" · [PubMed]({sel_r.pubmed_url})")
                    st.markdown(sel_r.abstract or "_No abstract available._")
                    st.download_button(
                        f"⬇ Download this abstract (PMID {sel_r.pmid})",
                        data=_format_single_abstract(sel_r),
                        file_name=f"cochrane_abstract_{sel_r.pmid}.txt",
                        mime="text/plain",
                    )

                st.subheader("Export Cochrane Reviews")
                rc1, rc2 = st.columns(2)
                rev_csv = io.StringIO()
                df_rev.to_csv(rev_csv, index=False)
                rc1.download_button("⬇ Download CSV", data=rev_csv.getvalue(),
                    file_name="cochrane_reviews.csv", mime="text/csv",
                    use_container_width=True)
                rc2.download_button("⬇ All Abstracts (TXT)",
                    data=_format_all_abstracts(c_reviews, saved_query),
                    file_name="cochrane_reviews_abstracts.txt", mime="text/plain",
                    use_container_width=True)
            else:
                st.info("No Cochrane Systematic Reviews found for this query.")

            st.divider()

            # ── CENTRAL — Controlled Trials ────────────────────────────────────
            st.subheader("CENTRAL — Controlled Trials")
            if c_central:
                with st.expander("PubMed query used"):
                    st.code(coch.get("central_query", ""), language="text")

                ct_rows = [{
                    "PMID":    r.pmid,
                    "Year":    r.pub_date[:4] if r.pub_date else "—",
                    "Title":   r.title,
                    "Authors": r.format_authors(2),
                    "Journal": r.journal,
                    "Type":    r.article_type[0] if r.article_type else "RCT",
                    "DOI":     r.doi or "",
                    "URL":     r.pubmed_url,
                } for r in c_central]
                df_ct = pd.DataFrame(ct_rows)
                st.dataframe(df_ct, use_container_width=True,
                    column_config={
                        "URL": st.column_config.LinkColumn("PubMed Link"),
                        "DOI": st.column_config.LinkColumn("DOI"),
                        "PMID": st.column_config.TextColumn("PMID", width="small"),
                        "Year": st.column_config.TextColumn("Year", width="small"),
                    }, hide_index=True)

                ct_sel = st.selectbox("Select a trial",
                    [f"{r.pmid} — {r.title[:65]}" for r in c_central],
                    key="selected_central_trial")
                if ct_sel:
                    sel_ct = next(r for r in c_central if ct_sel.startswith(r.pmid))
                    st.markdown(f"**{sel_ct.title}**")
                    st.caption(f"{sel_ct.format_authors(5)} · {sel_ct.journal} · "
                               f"{sel_ct.pub_date[:4]}" +
                               (f" · [DOI](https://doi.org/{sel_ct.doi})" if sel_ct.doi else "") +
                               f" · [PubMed]({sel_ct.pubmed_url})")
                    st.markdown(sel_ct.abstract or "_No abstract available._")
                    st.download_button(
                        f"⬇ Download this abstract (PMID {sel_ct.pmid})",
                        data=_format_single_abstract(sel_ct),
                        file_name=f"central_abstract_{sel_ct.pmid}.txt",
                        mime="text/plain",
                    )

                st.subheader("Export CENTRAL Trials")
                tc1, tc2 = st.columns(2)
                ct_csv = io.StringIO()
                df_ct.to_csv(ct_csv, index=False)
                tc1.download_button("⬇ Download CSV", data=ct_csv.getvalue(),
                    file_name="central_trials.csv", mime="text/csv",
                    use_container_width=True)
                tc2.download_button("⬇ All Abstracts (TXT)",
                    data=_format_all_abstracts(c_central, saved_query),
                    file_name="central_trials_abstracts.txt", mime="text/plain",
                    use_container_width=True)
            else:
                st.info("No controlled trials found for this query.")

        # ── Abstract viewer (PubMed) ───────────────────────────────────────────
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

        # ── New query button ───────────────────────────────────────────────────
        st.divider()
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            if st.button("🔄 Clear Results & Start New Query",
                         use_container_width=True, type="primary"):
                st.session_state.search_results   = None
                st.session_state.cochrane_results = None
                st.session_state.search_query     = ""
                st.rerun()

    else:
        st.warning("No articles found. Try broadening the query or removing date filters.")
