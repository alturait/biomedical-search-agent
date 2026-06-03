"""
Utility helpers: rate limiting, MeSH mapping, deduplication, export, Rich display.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


# ── Rate limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    """
    NCBI guideline: ≤3 req/s without API key, ≤10 req/s with API key.
    Default is conservative 2.5 req/s (0.4 s gap) regardless.
    """

    def __init__(self, requests_per_second: float = 2.5):
        self._min_gap = 1.0 / requests_per_second
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._min_gap:
            time.sleep(self._min_gap - elapsed)
        self._last = time.monotonic()


# ── Article data model ─────────────────────────────────────────────────────────

@dataclass
class ArticleResult:
    pmid: str
    title: str
    authors: list[str]
    journal: str
    pub_date: str
    abstract: str
    doi: Optional[str] = None
    article_type: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    pubmed_url: str = ""

    def __post_init__(self) -> None:
        if not self.pubmed_url:
            self.pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    def format_authors(self, max_authors: int = 3) -> str:
        if not self.authors:
            return "Unknown"
        shown = self.authors[:max_authors]
        suffix = f" et al. (+{len(self.authors) - max_authors})" if len(self.authors) > max_authors else ""
        return ", ".join(shown) + suffix

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArticleResult":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── MeSH vocabulary for wound care ────────────────────────────────────────────

WOUND_CARE_MESH: dict[str, str] = {
    # Wound healing & general
    "wound healing":           '"Wound Healing"[Mesh]',
    "wound care":              '"Wound Healing"[Mesh]',
    "chronic wound":           '("Chronic Disease"[Mesh] AND "Wounds and Injuries"[Mesh])',
    "wound management":        '"Wound Healing"[Mesh]',

    # Wound types
    "pressure ulcer":          '"Pressure Ulcer"[Mesh]',
    "pressure injury":         '"Pressure Ulcer"[Mesh]',
    "pressure sore":           '"Pressure Ulcer"[Mesh]',
    "decubitus":               '"Pressure Ulcer"[Mesh]',
    "bedsore":                 '"Pressure Ulcer"[Mesh]',
    "diabetic foot ulcer":     '"Diabetic Foot"[Mesh] AND "Foot Ulcer"[Mesh]',
    "diabetic foot":           '"Diabetic Foot"[Mesh]',
    "dfu":                     '"Diabetic Foot"[Mesh]',
    "venous leg ulcer":        '"Varicose Ulcer"[Mesh]',
    "venous ulcer":            '"Varicose Ulcer"[Mesh]',
    "leg ulcer":               '"Leg Ulcer"[Mesh]',
    "arterial ulcer":          '("Leg Ulcer"[Mesh] AND "Arterial Occlusive Diseases"[Mesh])',
    "surgical wound":          '"Surgical Wound"[Mesh]',
    "burn":                    '"Burns"[Mesh]',
    "skin tear":               '("Skin"[Mesh] AND "Lacerations"[Mesh])',
    "fistula":                 '"Fistula"[Mesh]',
    "pilonidal sinus":         '"Pilonidal Sinus"[Mesh]',

    # Negative pressure / VAC
    "negative pressure wound therapy": '"Negative-Pressure Wound Therapy"[Mesh]',
    "negative pressure":       '"Negative-Pressure Wound Therapy"[Mesh]',
    "npwt":                    '"Negative-Pressure Wound Therapy"[Mesh]',
    "vacuum assisted closure": '"Negative-Pressure Wound Therapy"[Mesh]',
    "vac therapy":             '"Negative-Pressure Wound Therapy"[Mesh]',

    # Dressings & materials
    "wound dressing":          '"Bandages"[Mesh]',
    "dressing":                '"Bandages"[Mesh]',
    "hydrocolloid":            '"Bandages, Hydrocolloid"[Mesh]',
    "occlusive dressing":      '"Occlusive Dressings"[Mesh]',
    "hydrogel":                '"Hydrogels"[Mesh]',
    "alginate":                '"Alginates"[Mesh]',
    "foam dressing":           '("Bandages"[Mesh] AND foam[tiab])',
    "silver dressing":         '("Silver"[Mesh] AND "Bandages"[Mesh])',
    "silver":                  '"Silver"[Mesh]',
    "antimicrobial dressing":  '("Anti-Infective Agents"[Mesh] AND "Bandages"[Mesh])',
    "collagen dressing":       '("Collagen"[Mesh] AND "Bandages"[Mesh])',
    "collagen":                '"Collagen"[Mesh]',
    "honey":                   '"Honey"[Mesh]',
    "iodine":                  '"Iodine"[Mesh]',
    "povidone iodine":         '"Povidone-Iodine"[Mesh]',

    # Procedures
    "debridement":             '"Debridement"[Mesh]',
    "skin graft":              '"Skin Transplantation"[Mesh]',
    "skin grafting":           '"Skin Transplantation"[Mesh]',
    "flap":                    '"Surgical Flaps"[Mesh]',
    "hyperbaric oxygen":       '"Hyperbaric Oxygenation"[Mesh]',
    "hbot":                    '"Hyperbaric Oxygenation"[Mesh]',
    "maggot therapy":          '("Larva"[Mesh] AND "Wound Healing"[Mesh])',
    "maggot":                  '("Larva"[Mesh] AND "Wound Healing"[Mesh])',
    "electrical stimulation":  '"Electric Stimulation Therapy"[Mesh]',
    "photobiomodulation":      '"Low-Level Light Therapy"[Mesh]',
    "laser therapy":           '"Low-Level Light Therapy"[Mesh]',
    "ultrasound therapy":      '"Ultrasonic Therapy"[Mesh]',
    "ultrasound":              '"Ultrasonic Therapy"[Mesh]',

    # Biologics & growth factors
    "platelet rich plasma":    '"Platelet-Rich Plasma"[Mesh]',
    "prp":                     '"Platelet-Rich Plasma"[Mesh]',
    "growth factor":           '"Intercellular Signaling Peptides and Proteins"[Mesh]',
    "stem cell":               '"Stem Cells"[Mesh]',
    "becaplermin":             '"Becaplermin"[Mesh]',
    "pdgf":                    '"Becaplermin"[Mesh]',

    # Infection / microbiology
    "wound infection":         '"Wound Infection"[Mesh]',
    "biofilm":                 '"Biofilms"[Mesh]',
    "mrsa":                    '"Methicillin-Resistant Staphylococcus aureus"[Mesh]',
    "antiseptic":              '"Anti-Infective Agents, Local"[Mesh]',
    "antibiotic":              '"Anti-Bacterial Agents"[Mesh]',

    # Patient populations / comorbidities
    "diabetes":                '"Diabetes Mellitus"[Mesh]',
    "elderly":                 '"Aged"[Mesh]',
    "older adults":            '"Aged"[Mesh]',
    "spinal cord injury":      '"Spinal Cord Injuries"[Mesh]',
    "obesity":                 '"Obesity"[Mesh]',
    "immunocompromised":       '"Immunocompromised Host"[Mesh]',
    "peripheral artery":       '"Peripheral Arterial Disease"[Mesh]',
    "lymphedema":              '"Lymphedema"[Mesh]',

    # Outcomes
    "quality of life":         '"Quality of Life"[Mesh]',
    "pain":                    '"Pain"[Mesh]',
    "cost effectiveness":      '"Cost-Benefit Analysis"[Mesh]',
    "healing rate":            "healing rate[tiab]",
    "wound closure":           "wound closure[tiab]",
    "recurrence":              '"Recurrence"[Mesh]',
    "amputation":              '"Amputation"[Mesh]',
}

ARTICLE_TYPE_FILTERS: dict[str, str] = {
    "all":              "",
    "review":           '"Review"[pt]',
    "systematic_review": '"Systematic Review"[pt]',
    "meta-analysis":    '"Meta-Analysis"[pt]',
    "rct":              '"Randomized Controlled Trial"[pt]',
    "clinical_trial":   '"Clinical Trial"[pt]',
    "observational":    '"Observational Study"[pt]',
    "guideline":        '"Guideline"[pt]',
    "case_report":      '"Case Reports"[pt]',
}


def build_mesh_query(
    natural_query: str,
    article_type: str = "all",
    date_range: Optional[tuple[str, str]] = None,
) -> str:
    """
    Map a natural language wound care query to an optimised PubMed search string.

    Strategy:
    1. Match known wound care terms (longest first to avoid partial conflicts).
    2. Fall back to title/abstract keyword search for unrecognised terms.
    3. Append article-type and date-range filters.
    """
    q_lower = natural_query.lower()
    mesh_parts: list[str] = []
    consumed: set[str] = set()

    # Match longest phrases first
    for term in sorted(WOUND_CARE_MESH, key=len, reverse=True):
        if term in q_lower:
            words = set(term.split())
            if not words.issubset(consumed):
                mesh_parts.append(WOUND_CARE_MESH[term])
                consumed |= words

    if mesh_parts:
        query = " AND ".join(f"({p})" for p in mesh_parts)
    else:
        # Plain keyword fallback on title/abstract
        tokens = [w for w in natural_query.split() if len(w) > 3][:6]
        query = " AND ".join(f'"{t}"[tiab]' for t in tokens) or natural_query

    # Article type
    type_filter = ARTICLE_TYPE_FILTERS.get(article_type.lower().replace(" ", "_"), "")
    if type_filter:
        query = f"({query}) AND {type_filter}"

    # Date range
    if date_range:
        start, end = date_range
        if start or end:
            start = start or "1900/01/01"
            end = end or datetime.now().strftime("%Y/%m/%d")
            query = f'({query}) AND ("{start}"[pdat]:"{end}"[pdat])'

    return query


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_results(results: list[ArticleResult]) -> list[ArticleResult]:
    seen_pmids: set[str] = set()
    seen_dois: set[str] = set()
    unique: list[ArticleResult] = []
    for r in results:
        if r.pmid in seen_pmids:
            continue
        if r.doi and r.doi in seen_dois:
            logger.debug("Duplicate DOI skipped: %s", r.doi)
            continue
        seen_pmids.add(r.pmid)
        if r.doi:
            seen_dois.add(r.doi)
        unique.append(r)
    return unique


# ── Export functions ───────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_to_csv(results: list[ArticleResult], filepath: str | Path) -> Path:
    fp = Path(filepath)
    rows = []
    for r in results:
        d = r.to_dict()
        d["authors"] = r.format_authors(max_authors=6)
        d["mesh_terms"] = "; ".join(r.mesh_terms[:8])
        d["article_type"] = "; ".join(r.article_type)
        rows.append(d)
    pd.DataFrame(rows).to_csv(fp, index=False)
    logger.info("CSV exported → %s", fp)
    return fp


def export_to_json(results: list[ArticleResult], filepath: str | Path) -> Path:
    fp = Path(filepath)
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in results], fh, indent=2, ensure_ascii=False)
    logger.info("JSON exported → %s", fp)
    return fp


def export_to_bibtex(results: list[ArticleResult], filepath: str | Path) -> Path:
    fp = Path(filepath)
    entries: list[str] = []
    for r in results:
        first_author = r.authors[0].split()[-1] if r.authors else "Unknown"
        year = r.pub_date[:4] if len(r.pub_date) >= 4 else "0000"
        key = f"{first_author}{year}_{r.pmid}"
        doi_line = f"  doi       = {{{r.doi}}}," if r.doi else ""
        authors_bib = " and ".join(r.authors) if r.authors else "Unknown"
        abstract_short = r.abstract[:500].replace("{", "").replace("}", "") + "..."
        entries.append(
            f"@article{{{key},\n"
            f"  author    = {{{authors_bib}}},\n"
            f"  title     = {{{{{r.title}}}}},\n"
            f"  journal   = {{{r.journal}}},\n"
            f"  year      = {{{year}}},\n"
            f"  pmid      = {{{r.pmid}}},\n"
            f"{doi_line}\n"
            f"  url       = {{{r.pubmed_url}}},\n"
            f"  abstract  = {{{abstract_short}}}\n"
            f"}}"
        )
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(entries))
    logger.info("BibTeX exported → %s", fp)
    return fp


# ── Rich display ───────────────────────────────────────────────────────────────

def display_results_table(results: list[ArticleResult], max_rows: int = 25) -> None:
    t = Table(
        title=f"PubMed Results  ({len(results)} articles)",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    t.add_column("#",        style="dim",     width=4)
    t.add_column("PMID",     style="cyan",    width=10)
    t.add_column("Year",     style="magenta", width=6)
    t.add_column("Type",     style="blue",    width=16)
    t.add_column("Authors",  style="green",   width=22)
    t.add_column("Title",    style="white",   width=46, no_wrap=False)
    t.add_column("Journal",  style="yellow",  width=22)

    for i, r in enumerate(results[:max_rows], 1):
        year = r.pub_date[:4] if r.pub_date else "—"
        art_type = (r.article_type[0] if r.article_type else "Article")[:14]
        title_trunc = r.title[:44] + "…" if len(r.title) > 45 else r.title
        journal_trunc = r.journal[:20] + "…" if len(r.journal) > 21 else r.journal
        t.add_row(str(i), r.pmid, year, art_type, r.format_authors(2), title_trunc, journal_trunc)

    console.print(t)
    if len(results) > max_rows:
        console.print(f"[dim]… and {len(results) - max_rows} more (exported to file)[/dim]")
