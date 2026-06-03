"""
Wound Care Literature Search Agent — CLI entry point.

Usage examples:
  python main.py "negative pressure wound therapy diabetic foot"
  python main.py "pressure ulcer prevention" -t systematic_review -n 30
  python main.py "best dressings pressure injuries 2024-2026" --date-from 2024/01/01 -o ./results
  python main.py --interactive
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

load_dotenv()

console = Console()
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("wound_care_agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── LLM factory ───────────────────────────────────────────────────────────────

def get_llm(provider: str, model: str | None = None):
    provider = provider.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(model=model or "gpt-4o", temperature=0, api_key=key)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(model=model or "claude-opus-4-7", api_key=key)

    if provider == "groq":
        from langchain_groq import ChatGroq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        return ChatGroq(model=model or "Qwen/Qwen3-32B", temperature=0, api_key=key)

    raise ValueError(f"Unknown provider: {provider!r}. Choose openai | anthropic | groq")


# ── Single search run ─────────────────────────────────────────────────────────

def run_search(
    query: str,
    *,
    provider: str = "openai",
    model: str | None = None,
    max_results: int = 20,
    article_type: str = "all",
    date_from: str = "",
    date_to: str = "",
    export_dir: str | None = None,
    verbose: bool = False,
    save_alert: bool = False,
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    console.print(Panel.fit(
        "[bold cyan]Wound Care Literature Search Agent[/bold cyan]\n"
        "[dim]PubMed · MeSH-optimised · LangGraph orchestration[/dim]",
        border_style="cyan",
    ))

    try:
        llm = get_llm(provider, model)
    except (EnvironmentError, ValueError) as exc:
        console.print(f"[red bold]Setup error:[/red bold] {exc}")
        sys.exit(1)

    from agent import WoundCareAgent
    from utils import display_results_table

    ncbi_key = os.getenv("NCBI_API_KEY", "")
    agent = WoundCareAgent(llm, ncbi_api_key=ncbi_key)

    console.print(f"\n[yellow]Query:[/yellow] {query}")
    console.print(
        f"[dim]max={max_results}  type={article_type}  "
        f"from={date_from or 'any'}  to={date_to or 'any'}[/dim]\n"
    )

    with console.status("[cyan]Searching PubMed…[/cyan]", spinner="dots"):
        result = agent.search(
            query=query,
            max_results=max_results,
            article_type=article_type,
            date_from=date_from,
            date_to=date_to,
            export_dir=export_dir,
        )

    articles = result["articles"]
    meta     = result["search_metadata"]

    # Stats
    console.print(
        f"[green]Found:[/green] {meta.get('total_count', '?')} total results  |  "
        f"Retrieved: {len(articles)} articles\n"
        f"[dim]Query used: {meta.get('query_used', 'N/A')}[/dim]\n"
    )

    if articles:
        display_results_table(articles)
    else:
        console.print("[yellow]No articles returned. Try broadening the query.[/yellow]")

    if result["summary"]:
        console.print()
        console.print(Panel(
            Markdown(result["summary"]),
            title="[bold green]Clinical Evidence Summary[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

    if result["export_paths"]:
        console.print("\n[bold]Exported files:[/bold]")
        for fmt, path in result["export_paths"].items():
            console.print(f"  [cyan]{fmt.upper():8}[/cyan] {path}")

    if save_alert or Confirm.ask("\nSave as a search alert?", default=False):
        alert_name = Prompt.ask("Alert name", default=query[:40])
        agent.save_alert(alert_name, query, {
            "max_results":  max_results,
            "article_type": article_type,
            "date_from":    date_from,
            "date_to":      date_to,
        })


# ── Interactive REPL ──────────────────────────────────────────────────────────

def run_interactive(provider: str = "openai", model: str | None = None) -> None:
    console.print(Panel.fit(
        "[bold cyan]Wound Care Literature Search — Interactive Mode[/bold cyan]\n"
        "[dim]Type 'quit' or 'exit' to stop[/dim]",
        border_style="cyan",
    ))

    try:
        llm = get_llm(provider, model)
    except (EnvironmentError, ValueError) as exc:
        console.print(f"[red bold]Setup error:[/red bold] {exc}")
        return

    from agent import WoundCareAgent
    from utils import display_results_table

    ncbi_key = os.getenv("NCBI_API_KEY", "")
    agent = WoundCareAgent(llm, ncbi_api_key=ncbi_key)

    export_dir = os.getenv("EXPORT_DIR", "./results")

    while True:
        console.print("\n[dim]" + "─" * 60 + "[/dim]")
        query = Prompt.ask("[cyan]Search query[/cyan]").strip()
        if query.lower() in {"quit", "exit", "q", ""}:
            console.print("[yellow]Goodbye![/yellow]")
            break

        max_results  = int(Prompt.ask("Max results", default="20"))
        article_type = Prompt.ask(
            "Article type",
            choices=["all", "review", "systematic_review", "meta-analysis",
                     "rct", "clinical_trial", "guideline", "case_report"],
            default="all",
        )
        date_from = Prompt.ask("Date from (YYYY/MM/DD or blank)", default="")
        date_to   = Prompt.ask("Date to   (YYYY/MM/DD or blank)", default="")
        do_export = Confirm.ask("Export to CSV/JSON/BibTeX?", default=False)

        with console.status("[cyan]Working…[/cyan]", spinner="dots"):
            result = agent.search(
                query=query,
                max_results=max_results,
                article_type=article_type,
                date_from=date_from,
                date_to=date_to,
                export_dir=export_dir if do_export else None,
            )

        articles = result["articles"]
        meta     = result["search_metadata"]

        console.print(
            f"\n[green]Retrieved {len(articles)} articles[/green]  "
            f"(total hits: {meta.get('total_count','?')})"
        )

        if articles:
            display_results_table(articles)

        if result["summary"]:
            console.print(Panel(
                Markdown(result["summary"]),
                title="[bold green]Evidence Summary[/bold green]",
                border_style="green",
                padding=(1, 2),
            ))

        if result.get("export_paths"):
            for fmt, path in result["export_paths"].items():
                console.print(f"[cyan]{fmt.upper()}[/cyan] → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="wound-care-agent",
        description="PubMed literature search agent for wound care topics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "negative pressure wound therapy diabetic foot"
  python main.py "pressure ulcer prevention" -t systematic_review -n 30
  python main.py "wound dressings 2024" --date-from 2024/01/01 --date-to 2026/12/31 -o ./results
  python main.py --interactive -p anthropic -m claude-opus-4-7
        """,
    )

    p.add_argument("query", nargs="?", help="Natural language search query")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="Start interactive REPL session")

    # Search params
    p.add_argument("--max-results", "-n", type=int, default=20,
                   metavar="N", help="Max articles to retrieve (default: 20)")
    p.add_argument("--article-type", "-t", default="all",
                   choices=["all", "review", "systematic_review", "meta-analysis",
                            "rct", "clinical_trial", "observational", "guideline", "case_report"],
                   metavar="TYPE", help="Filter by study design")
    p.add_argument("--date-from", default="", metavar="YYYY/MM/DD",
                   help="Earliest publication date")
    p.add_argument("--date-to", default="", metavar="YYYY/MM/DD",
                   help="Latest publication date")

    # Output
    p.add_argument("--export-dir", "-o", default=None, metavar="DIR",
                   help="Directory for CSV / JSON / BibTeX export")
    p.add_argument("--save-alert", action="store_true",
                   help="Automatically save search as an alert without prompting")

    # LLM
    p.add_argument("--provider", "-p", default=os.getenv("LLM_PROVIDER", "openai"),
                   choices=["openai", "anthropic", "groq"],
                   help="LLM provider (default: openai)")
    p.add_argument("--model", "-m", default=os.getenv("LLM_MODEL"),
                   help="Override model name")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable INFO-level logging")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.interactive:
        run_interactive(provider=args.provider, model=args.model)
    elif args.query:
        run_search(
            args.query,
            provider=args.provider,
            model=args.model,
            max_results=args.max_results,
            article_type=args.article_type,
            date_from=args.date_from,
            date_to=args.date_to,
            export_dir=args.export_dir,
            verbose=args.verbose,
            save_alert=args.save_alert,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
