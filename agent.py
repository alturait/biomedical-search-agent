"""
LangGraph agent orchestrating the wound care literature search workflow.

Workflow:
  1. agent node  – LLM decides which tool to call next
  2. tools node  – executes the chosen tool
  3. summarize   – structured clinical evidence summary
  4. END

State accumulates PMIDs and article dicts across multiple tool calls so the
final summarise step always sees the full result set.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from tools import fetch_abstracts_tool, refine_query_tool, search_pubmed_tool
from utils import (
    ArticleResult,
    classify_evidence_level,
    console,
    deduplicate_results,
    display_results_table,
    export_to_bibtex,
    export_to_csv,
    export_to_json,
    rank_articles,
)

logger = logging.getLogger(__name__)

TOOLS = [refine_query_tool, search_pubmed_tool, fetch_abstracts_tool]

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert biomedical literature search assistant specialising in wound care.

WORKFLOW (follow this order every time):
1. Call refine_wound_care_query to map the user's query to MeSH terms.
2. Call search_pubmed with the refined query to obtain PMIDs.
3. Call fetch_abstracts with those PMIDs to get titles, authors, abstracts, and MeSH terms.
4. After fetching, stop calling tools — the summarise step will handle synthesis.

SEARCH STRATEGY:
• Default to use_mesh=True unless the user provides a raw PubMed query.
• For high-evidence queries use article_type="systematic_review" or "meta-analysis".
• For recent evidence add date_from / date_to parameters.
• Retrieve at least 10 articles unless the user specifies otherwise.

CLINICAL SCOPE:
Wound types: pressure ulcers/injuries, diabetic foot ulcers, venous/arterial leg ulcers,
surgical wounds, burns, traumatic wounds.
Treatments: NPWT, dressings (hydrocolloid, foam, alginate, silver, hydrogel), debridement,
HBOT, electrical stimulation, biologics (PRP, growth factors, stem cells), skin grafts.
Outcomes: healing rate, wound closure, amputation, QoL, pain, cost-effectiveness.
"""

_SUMMARY_SYSTEM = """\
You are a senior wound care clinician preparing a structured evidence summary for a clinical team.
Be concise, precise, and grounded in the evidence provided. Use plain language where possible.
"""

_APPEAL_SYSTEM = """\
You are assisting a physician's office in drafting the clinical-evidence section of a prior \
authorization appeal for a wound-care cellular/tissue-based product (CTP). Ground every claim \
strictly in the candidate articles provided — never invent citations, statistics, or claims not \
supported by the given abstracts. This section is evidence-only and must not reference any \
specific patient by name or identifying detail. Be concise and professional.
"""

# Denial-reason-specific framing — this is what makes the rationale rebut the
# actual denial rather than restate generic product benefits (generic appeal
# letters underperform targeted ones).
_DENIAL_REASON_FRAMING: dict[str, str] = {
    "not_medically_necessary": (
        "The payer denied this request as 'not medically necessary.' Frame the rationale around "
        "objective clinical criteria (wound chronicity, documented failure of conservative care) "
        "and the peer-reviewed evidence base demonstrating clinical efficacy for this indication."
    ),
    "investigational_experimental": (
        "The payer denied this request as 'investigational' or 'experimental.' Frame the rationale "
        "around the existence of peer-reviewed, published clinical evidence (RCTs / systematic "
        "reviews where available) establishing this as an evidence-based treatment rather than an "
        "experimental one. Do not assert FDA clearance/approval status unless it is stated in the "
        "provided abstracts."
    ),
    "conservative_care_not_exhausted": (
        "The payer denied this request stating conservative/standard wound care was not exhausted. "
        "Assume the conservative-care trial and its failure are already documented elsewhere in the "
        "appeal — do not restate them. Frame the rationale around why the evidence supports "
        "escalating to this product once standard care has failed."
    ),
    "other": (
        "Frame the rationale as a general medical-necessity argument grounded strictly in the "
        "evidence provided."
    ),
}


class _AppealKeyFinding(BaseModel):
    pmid: str = Field(description="PubMed ID of the cited article — must be one of the candidate PMIDs given.")
    key_finding: str = Field(description="One sentence summarizing this article's finding as it supports the appeal.")


class _AppealEvidence(BaseModel):
    clinical_rationale: str = Field(
        description="2-4 sentence clinical rationale paragraph, professional tone, suitable for a "
                    "physician-signed prior authorization appeal letter."
    )
    key_findings: list[_AppealKeyFinding] = Field(
        description="One entry per candidate article provided, same PMIDs, in the same order."
    )


# ── Agent state ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:        Annotated[list[BaseMessage], add_messages]
    query:           str
    pmids:           list[str]
    articles:        list[dict]
    summary:         str
    search_metadata: dict
    export_paths:    dict


# ── Helper: extract tool outputs from message history ─────────────────────────

def _extract_from_messages(messages: list[BaseMessage]) -> tuple[list[str], list[dict], dict]:
    """Walk tool-result messages and accumulate pmids, articles, and search metadata."""
    pmids: list[str] = []
    articles: list[dict] = []
    meta: dict = {}

    for msg in messages:
        # ToolMessages carry the serialised tool output as their content
        if not hasattr(msg, "content"):
            continue
        content = msg.content
        if not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue

        if data.get("pmids"):
            pmids = data["pmids"]
            meta = {
                "total_count": data.get("total_count", 0),
                "query_used":  data.get("query_used", ""),
            }
        if data.get("articles"):
            articles = data["articles"]

    return pmids, articles, meta


# ── Node implementations ───────────────────────────────────────────────────────

class WoundCareAgent:
    """End-to-end wound care literature search agent."""

    def __init__(self, llm: BaseChatModel, ncbi_api_key: str = "") -> None:
        self.llm = llm.bind_tools(TOOLS)
        self.llm_plain = llm        # un-bound, for summarisation
        self.ncbi_api_key = ncbi_api_key
        self._tool_node = ToolNode(TOOLS)
        self.graph = self._build_graph()

    # ── Graph construction ─────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("agent",     self._agent_node)
        g.add_node("tools",     self._tool_node)
        g.add_node("summarize", self._summarize_node)

        g.set_entry_point("agent")
        g.add_conditional_edges("agent", self._route, {
            "tools":     "tools",
            "summarize": "summarize",
            "end":       END,
        })
        g.add_edge("tools",     "agent")
        g.add_edge("summarize", END)
        return g.compile()

    # ── agent node ─────────────────────────────────────────────────────────────

    def _agent_node(self, state: AgentState) -> dict:
        msgs = [SystemMessage(content=_SYSTEM)] + state["messages"]
        response = self.llm.invoke(msgs)

        pmids, articles, meta = _extract_from_messages(state["messages"])
        # Prefer accumulated state over newly extracted (state may already be populated)
        pmids    = state.get("pmids") or pmids
        articles = state.get("articles") or articles
        meta     = state.get("search_metadata") or meta

        return {
            "messages":        [response],
            "pmids":           pmids,
            "articles":        articles,
            "search_metadata": meta,
        }

    # ── routing ────────────────────────────────────────────────────────────────

    def _route(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        # Articles fetched but not yet summarised → go to summarise
        if state.get("articles") and not state.get("summary"):
            return "summarize"
        return "end"

    # ── summarise node ─────────────────────────────────────────────────────────

    def _summarize_node(self, state: AgentState) -> dict:
        articles = state.get("articles", [])
        if not articles:
            msg = "No articles retrieved — nothing to summarise."
            return {"summary": msg, "messages": [AIMessage(content=msg)]}

        query = state.get("query", "wound care")
        meta  = state.get("search_metadata", {})

        # Build a compact but rich abstract block (cap at 20 articles for context)
        blocks: list[str] = []
        for i, a in enumerate(articles[:20], 1):
            authors = a.get("authors", [])
            fa = authors[0] if authors else "Unknown"
            year = a.get("pub_date", "")[:4] or "?"
            types = ", ".join(a.get("article_type", [])[:2]) or "Article"
            abstract_snippet = (a.get("abstract") or "No abstract")[:700]
            mesh_snippet = "; ".join(a.get("mesh_terms", [])[:6])

            blocks.append(
                f"[{i}] PMID {a.get('pmid','?')} | {types} | {fa} {year}\n"
                f"Title: {a.get('title','?')}\n"
                f"Journal: {a.get('journal','?')}\n"
                f"MeSH: {mesh_snippet}\n"
                f"Abstract: {abstract_snippet}…\n"
            )

        prompt = (
            f"The following {len(articles)} articles were retrieved from PubMed "
            f"for the query: \"{query}\"\n"
            f"(Search returned {meta.get('total_count','?')} total hits; "
            f"query used: {meta.get('query_used','?')})\n\n"
            + "─" * 60 + "\n"
            + ("\n" + "─" * 40 + "\n").join(blocks)
            + "\n" + "─" * 60 + "\n\n"
            "Please produce a **structured clinical evidence summary** with these sections:\n\n"
            "## Executive Summary\n"
            "3-4 sentences: current state of evidence, consensus, and key uncertainty.\n\n"
            "## Key Findings\n"
            "Bullet list of the most important, clinically actionable findings.\n\n"
            "## Evidence Quality\n"
            "Breakdown by study design; note sample sizes and risk of bias where apparent.\n\n"
            "## Clinical Implications\n"
            "What should clinicians do differently based on this evidence?\n\n"
            "## Gaps & Emerging Research\n"
            "What is still unknown? What emerging treatments/approaches appear promising?\n\n"
            "## Top Recommended Articles\n"
            "List 3-5 PMIDs with one-sentence justification each.\n"
        )

        response = self.llm_plain.invoke([
            SystemMessage(content=_SUMMARY_SYSTEM),
            HumanMessage(content=prompt),
        ])
        summary = response.content if hasattr(response, "content") else str(response)

        return {"summary": summary, "messages": [AIMessage(content=summary)]}

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        max_results: int = 20,
        article_type: str = "all",
        date_from: str = "",
        date_to: str = "",
        export_dir: Optional[str] = None,
    ) -> dict:
        """
        Run the full search workflow and return a result dict with keys:
          articles (list[ArticleResult]), summary (str), pmids (list[str]),
          search_metadata (dict), export_paths (dict).
        """
        user_msg = HumanMessage(content=(
            f"Search PubMed for: {query}\n"
            f"Parameters: max_results={max_results}, article_type={article_type}, "
            f"date_from={date_from or 'none'}, date_to={date_to or 'none'}.\n\n"
            "Steps: 1) refine_wound_care_query → 2) search_pubmed → 3) fetch_abstracts."
        ))

        init: AgentState = {
            "messages":        [user_msg],
            "query":           query,
            "pmids":           [],
            "articles":        [],
            "summary":         "",
            "search_metadata": {},
            "export_paths":    {},
        }

        final = self.graph.invoke(init)

        # Convert raw dicts back to ArticleResult objects and deduplicate
        article_objects = [
            ArticleResult.from_dict(a)
            for a in final.get("articles", [])
            if a
        ]
        article_objects = deduplicate_results(article_objects)

        export_paths: dict = {}
        if export_dir and article_objects:
            export_paths = self._export(article_objects, query, export_dir)

        return {
            "articles":        article_objects,
            "summary":         final.get("summary", ""),
            "pmids":           final.get("pmids", []),
            "search_metadata": final.get("search_metadata", {}),
            "export_paths":    export_paths,
        }

    def _export(self, articles: list[ArticleResult], query: str, export_dir: str) -> dict:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() else "_" for c in query[:30])
        base = f"wound_care_{slug}_{ts}"
        out  = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: dict = {}
        try:
            paths["csv"]    = str(export_to_csv(articles,    out / f"{base}.csv"))
            paths["json"]   = str(export_to_json(articles,   out / f"{base}.json"))
            paths["bibtex"] = str(export_to_bibtex(articles, out / f"{base}.bib"))
        except Exception as exc:
            logger.error("Export error: %s", exc)
        return paths

    def generate_appeal_evidence(
        self,
        *,
        product: str,
        diagnosis: str,
        denial_reason: str,
        max_results: int = 15,
    ) -> dict:
        """
        Search PubMed for the strongest evidence supporting `product` for `diagnosis`,
        then draft a denial-reason-framed clinical rationale plus a PMID evidence table
        for Section 4 of a wound-graft prior-authorization appeal.

        Inputs are limited to product / diagnosis / denial-reason category — no
        patient-identifying data is accepted or produced by this method.

        Returns:
          {
            "clinical_rationale": str,
            "evidence_rows": [{"pmid", "citation", "evidence_level", "key_finding", "pubmed_url"}, ...],
            "articles_considered": int,
            "search_metadata": dict,
          }
        """
        query = f"{product} {diagnosis}"
        search_result = self.search(query=query, max_results=max_results)
        articles = search_result["articles"]

        if not articles:
            return {
                "clinical_rationale": "",
                "evidence_rows": [],
                "articles_considered": 0,
                "search_metadata": search_result["search_metadata"],
            }

        ranked = rank_articles(articles)[:3]
        framing = _DENIAL_REASON_FRAMING.get(denial_reason, _DENIAL_REASON_FRAMING["other"])

        candidates_block = "\n".join(
            f"- PMID {a.pmid} | {classify_evidence_level(a.article_type)} | "
            f"{a.format_authors(3)} ({(a.pub_date or '')[:4]}) | {a.journal}\n"
            f"  Title: {a.title}\n"
            f"  Abstract: {(a.abstract or 'No abstract')[:600]}"
            for a in ranked
        )

        prompt = (
            f"Product/CTP under appeal: {product}\n"
            f"Diagnosis: {diagnosis}\n"
            f"Denial reason category: {denial_reason}\n\n"
            f"{framing}\n\n"
            "Candidate supporting articles (already selected as the strongest available "
            "evidence for this product/diagnosis — cite ONLY these, do not invent PMIDs or "
            "claims beyond what is stated in the abstracts below):\n\n"
            f"{candidates_block}\n\n"
            "Write the clinical rationale paragraph and, for EACH candidate article above, "
            "a one-sentence key finding relevant to this appeal."
        )

        structured_llm = self.llm_plain.with_structured_output(_AppealEvidence)
        result: _AppealEvidence = structured_llm.invoke([
            SystemMessage(content=_APPEAL_SYSTEM),
            HumanMessage(content=prompt),
        ])

        finding_by_pmid = {kf.pmid: kf.key_finding for kf in result.key_findings}
        evidence_rows = [
            {
                "pmid":           a.pmid,
                "citation":       f"{a.format_authors(3)}, {a.journal}, {(a.pub_date or '')[:4]}",
                "evidence_level": classify_evidence_level(a.article_type),
                "key_finding":    finding_by_pmid.get(a.pmid, "See abstract."),
                "pubmed_url":     a.pubmed_url,
            }
            for a in ranked
        ]

        return {
            "clinical_rationale": result.clinical_rationale,
            "evidence_rows":      evidence_rows,
            "articles_considered": len(articles),
            "search_metadata":    search_result["search_metadata"],
        }

    def save_alert(self, name: str, query: str, params: dict, filepath: str = "alerts.json") -> None:
        """Persist a saved search for later re-running."""
        fp = Path(filepath)
        alerts: list[dict] = []
        if fp.exists():
            with open(fp) as fh:
                alerts = json.load(fh)

        alerts.append({
            "name":     name,
            "query":    query,
            "params":   params,
            "created":  datetime.now().isoformat(),
            "last_run": None,
        })
        with open(fp, "w") as fh:
            json.dump(alerts, fh, indent=2)
        console.print(f"[green]Search alert saved → {fp}[/green]")
