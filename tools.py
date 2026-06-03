"""
LangChain tools wrapping PubMed E-utilities (ESearch + EFetch + query refinement).

Designed to be importable by both the LangGraph agent and standalone scripts.
Extensible: add Cochrane / PMC / Google Scholar tools in the same pattern.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Optional

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils import (
    RateLimiter,
    ArticleResult,
    ARTICLE_TYPE_FILTERS,
    WOUND_CARE_MESH,
    build_mesh_query,
)

logger = logging.getLogger(__name__)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_rate = RateLimiter(requests_per_second=2.5)

# ── Input schemas (Pydantic v2) ────────────────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(
        description="Natural language wound care query or raw PubMed search string."
    )
    max_results: int = Field(
        default=20, ge=1, le=200,
        description="Maximum number of articles to return (1-200).",
    )
    article_type: str = Field(
        default="all",
        description=(
            "Filter by study design. Choices: all | review | systematic_review | "
            "meta-analysis | rct | clinical_trial | observational | guideline | case_report"
        ),
    )
    date_from: str = Field(
        default="",
        description="Earliest publication date, format YYYY/MM/DD (e.g. 2022/01/01). Leave blank for no limit.",
    )
    date_to: str = Field(
        default="",
        description="Latest publication date, format YYYY/MM/DD. Leave blank for no limit.",
    )
    use_mesh: bool = Field(
        default=True,
        description="Automatically map wound care terms to MeSH controlled vocabulary for precision.",
    )
    api_key: str = Field(
        default="",
        description="NCBI API key (optional). Raises rate limit from 3 → 10 req/s.",
    )


class FetchInput(BaseModel):
    pmids: list[str] = Field(
        description="List of PubMed IDs (strings) to retrieve full details for."
    )
    api_key: str = Field(default="", description="NCBI API key (optional).")


class RefineInput(BaseModel):
    query: str = Field(
        description="Natural language wound care query to analyse for MeSH term mapping."
    )


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """Rate-limited GET with retry on transient errors."""
    _rate.wait()
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _robust_get(url: str, params: dict, timeout: int = 60) -> requests.Response:
    return _get(url, params, timeout)


# ── ESearch ───────────────────────────────────────────────────────────────────

def _esearch(
    query: str,
    max_results: int,
    date_from: str,
    date_to: str,
    api_key: str,
) -> tuple[list[str], int]:
    params: dict = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
    }
    if date_from:
        params.update({"mindate": date_from.replace("/", ""), "datetype": "pdat"})
    if date_to:
        params.update({"maxdate": date_to.replace("/", ""), "datetype": "pdat"})
    if api_key:
        params["api_key"] = api_key

    data = _robust_get(f"{NCBI_BASE}/esearch.fcgi", params).json()
    result = data.get("esearchresult", {})
    pmids = result.get("idlist", [])
    count = int(result.get("count", 0))
    logger.info("ESearch: %d total hits, returning %d PMIDs  |  query=%r", count, len(pmids), query[:80])
    return pmids, count


# ── EFetch + XML parsing ───────────────────────────────────────────────────────

def _efetch_batch(pmids: list[str], api_key: str) -> list[ArticleResult]:
    params: dict = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key
    resp = _robust_get(f"{NCBI_BASE}/efetch.fcgi", params, timeout=90)
    return _parse_pubmed_xml(resp.text)


def _parse_pubmed_xml(xml_text: str) -> list[ArticleResult]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("XML parse error: %s", exc)
        return []

    results: list[ArticleResult] = []
    for elem in root.findall(".//PubmedArticle"):
        try:
            art = _parse_article(elem)
            if art:
                results.append(art)
        except Exception as exc:
            logger.warning("Skipping malformed article element: %s", exc)
    return results


def _parse_article(elem: ET.Element) -> Optional[ArticleResult]:
    pmid_elem = elem.find(".//PMID")
    if pmid_elem is None or not pmid_elem.text:
        return None
    pmid = pmid_elem.text.strip()

    # Title (may contain inline XML like <i>, <b>, <sub>, <sup>)
    title_elem = elem.find(".//ArticleTitle")
    title = ("".join(title_elem.itertext()).strip().rstrip(".") if title_elem is not None else "No title")

    # Authors
    authors: list[str] = []
    for auth in elem.findall(".//Author"):
        last = auth.findtext("LastName", "").strip()
        fore = auth.findtext("ForeName", "").strip()
        if last:
            authors.append(f"{last} {fore}".strip())

    # Journal
    journal = (
        elem.findtext(".//Journal/Title")
        or elem.findtext(".//MedlineJournalInfo/MedlineTA")
        or "Unknown"
    )

    # Publication date
    pub_date = _extract_date(elem)

    # Abstract (structured or plain)
    abstract_parts: list[str] = []
    for atext in elem.findall(".//AbstractText"):
        label = atext.get("Label", "")
        text = "".join(atext.itertext()).strip()
        if label:
            abstract_parts.append(f"{label}: {text}")
        elif text:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    # DOI
    doi: Optional[str] = None
    for aid in elem.findall(".//ArticleId"):
        if aid.get("IdType") == "doi" and aid.text:
            doi = aid.text.strip()
            break

    # Publication types (filter boilerplate)
    _skip = {"Journal Article", "English Abstract"}
    article_types = [
        pt.text
        for pt in elem.findall(".//PublicationTypeList/PublicationType")
        if pt.text and pt.text not in _skip
    ]

    # MeSH descriptors
    mesh_terms = [
        mh.findtext("DescriptorName", "")
        for mh in elem.findall(".//MeshHeadingList/MeshHeading")
        if mh.findtext("DescriptorName")
    ]

    return ArticleResult(
        pmid=pmid,
        title=title,
        authors=authors,
        journal=journal,
        pub_date=pub_date,
        abstract=abstract,
        doi=doi,
        article_type=article_types,
        mesh_terms=mesh_terms,
    )


def _extract_date(elem: ET.Element) -> str:
    pub = elem.find(".//JournalIssue/PubDate")
    if pub is not None:
        year  = pub.findtext("Year", "")
        month = pub.findtext("Month", "01")
        day   = pub.findtext("Day", "01")
        ml    = pub.findtext("MedlineDate", "")
        if year:
            return f"{year}/{month}/{day}"
        if ml:
            return ml[:4]

    art_date = elem.find(".//ArticleDate")
    if art_date is not None:
        year  = art_date.findtext("Year", "")
        month = art_date.findtext("Month", "01")
        day   = art_date.findtext("Day", "01")
        if year:
            return f"{year}/{month}/{day}"

    return "Unknown"


# ── LangChain tools ────────────────────────────────────────────────────────────

@tool("search_pubmed", args_schema=SearchInput)
def search_pubmed_tool(
    query: str,
    max_results: int = 20,
    article_type: str = "all",
    date_from: str = "",
    date_to: str = "",
    use_mesh: bool = True,
    api_key: str = "",
) -> dict:
    """
    Search PubMed for biomedical literature on wound care topics.

    Converts natural language to an optimised MeSH-based PubMed query, runs
    ESearch, and returns a list of PMIDs plus the query actually used.
    Call this first, then use fetch_abstracts to retrieve article details.
    """
    try:
        if use_mesh:
            date_range = (date_from, date_to) if (date_from or date_to) else None
            refined = build_mesh_query(query, article_type, date_range)
        else:
            refined = query

        pmids, total = _esearch(refined, max_results, date_from, date_to, api_key)
        return {
            "success":        True,
            "pmids":          pmids,
            "total_count":    total,
            "returned_count": len(pmids),
            "query_used":     refined,
            "original_query": query,
        }
    except requests.RequestException as exc:
        logger.error("PubMed ESearch failed: %s", exc)
        return {"success": False, "error": str(exc), "pmids": [], "total_count": 0}


@tool("fetch_abstracts", args_schema=FetchInput)
def fetch_abstracts_tool(pmids: list[str], api_key: str = "") -> dict:
    """
    Fetch full article metadata and abstracts for a list of PubMed IDs.

    Processes up to 200 PMIDs in batches of 20 to comply with NCBI guidelines.
    Returns structured article records including title, authors, journal, abstract,
    MeSH terms, DOI, and direct PubMed URL.
    """
    if not pmids:
        return {"success": False, "error": "No PMIDs provided", "articles": [], "count": 0}

    all_results: list[ArticleResult] = []
    batch_size = 20

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        try:
            batch_results = _efetch_batch(batch, api_key)
            all_results.extend(batch_results)
            logger.info("Fetched batch %d/%d  (%d articles)", i // batch_size + 1,
                        -(-len(pmids) // batch_size), len(batch_results))
        except requests.RequestException as exc:
            logger.error("Batch EFetch failed for PMIDs %s…: %s", batch[:3], exc)

    return {
        "success":  True,
        "articles": [r.to_dict() for r in all_results],
        "count":    len(all_results),
    }


@tool("refine_wound_care_query", args_schema=RefineInput)
def refine_query_tool(query: str) -> dict:
    """
    Analyse a natural language wound care query and return the best PubMed search
    string, the matched MeSH concepts, and suggestions for improving recall/precision.

    Always call this before search_pubmed to understand how the query will be mapped.
    """
    q_lower = query.lower()
    matched: dict[str, str] = {}

    for term in sorted(WOUND_CARE_MESH, key=len, reverse=True):
        if term in q_lower:
            matched[term] = WOUND_CARE_MESH[term]

    refined = build_mesh_query(query)

    suggestions: list[str] = []
    if not matched:
        suggestions += [
            "No known wound care MeSH terms detected. Consider adding: "
            "wound type (e.g. pressure ulcer, diabetic foot), treatment "
            "(e.g. NPWT, dressing), or outcome (e.g. healing rate, amputation).",
            "Query will fall back to title/abstract keyword search — lower precision.",
        ]
    if len(matched) > 4:
        suggestions.append(
            "Many MeSH terms detected. Consider narrowing the query for higher precision."
        )

    return {
        "original_query":    query,
        "refined_query":     refined,
        "matched_mesh_terms": matched,
        "suggestions":       suggestions,
        "available_article_type_filters": list(ARTICLE_TYPE_FILTERS.keys()),
        "tip": "Pass refined_query directly to search_pubmed as the query parameter with use_mesh=False for exact control.",
    }
