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

from tools import fetch_abstracts_tool, refine_query_tool, search_pubmed_tool
from utils import (
    ArticleResult,
    console,
    deduplicate_results,
    display_results_table,
    export_to_bibtex,
    export_to_csv,
    export_to_json,
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
