"""
Search query audit logger.

Writes one record per search to:
  1. Python's standard logger → stdout → visible in Render's Log Stream
  2. search_log.csv in the working directory (persistent locally;
     ephemeral on Render between deploys unless a persistent disk is attached)
"""
from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_FILE = Path(os.getenv("SEARCH_LOG_FILE", "search_log.csv"))

_FIELDS = [
    "timestamp",
    "email",
    "query",
    "provider",
    "model",
    "article_type",
    "date_from",
    "date_to",
    "pubmed_results",
    "cochrane_reviews",
    "central_trials",
]


def log_search(
    *,
    email: str,
    query: str,
    provider: str,
    model: str,
    article_type: str,
    date_from: str,
    date_to: str,
    pubmed_results: int,
    cochrane_reviews: int = 0,
    central_trials: int = 0,
) -> None:
    """Log a completed search event."""
    record = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "email":           email,
        "query":           query,
        "provider":        provider,
        "model":           model,
        "article_type":    article_type,
        "date_from":       date_from or "",
        "date_to":         date_to or "",
        "pubmed_results":  pubmed_results,
        "cochrane_reviews": cochrane_reviews,
        "central_trials":  central_trials,
    }

    # Structured log line → Render Log Stream (searchable, downloadable)
    logger.info("SEARCH_EVENT %s", json.dumps(record))

    # Append to CSV
    try:
        write_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(record)
    except OSError as exc:
        logger.warning("Could not write to %s: %s", LOG_FILE, exc)
