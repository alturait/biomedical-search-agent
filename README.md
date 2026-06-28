# Wound Care Literature Search Agent

A production-ready PubMed search agent for wound care clinicians and researchers.
Uses LangGraph for orchestration, NCBI E-utilities for literature retrieval, and an
LLM of your choice to generate structured clinical evidence summaries.

---

## Features

| Feature | Detail |
|---------|--------|
| **MeSH mapping** | 70+ wound care terms auto-mapped to PubMed controlled vocabulary |
| **Study design filters** | RCT, systematic review, meta-analysis, guideline, case report, … |
| **Date range** | Filter by publication date (ESearch `pdat`) |
| **Structured summaries** | Executive summary · Key findings · Evidence quality · Clinical implications · Gaps |
| **Export** | CSV · JSON · BibTeX in one command |
| **Saved searches** | Persist query params to `alerts.json` for re-running |
| **Streamlit UI** | Optional web interface with abstract viewer and in-browser downloads |
| **Extensible** | Add Cochrane / PMC / Google Scholar sources by adding new tools in `tools.py` |

---

## Quick start

### 1 — Clone / copy the project

```
wound_care_agent/
├── agent.py          # LangGraph agent (search → fetch → summarise)
├── tools.py          # PubMed ESearch + EFetch LangChain tools
├── utils.py          # MeSH map, rate limiter, export, Rich display
├── main.py           # CLI entry point
├── streamlit_app.py  # Optional web UI
├── requirements.txt
├── .env.example      # Copy to .env and fill in keys
└── README.md
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Only install the LLM provider package you intend to use:

```bash
pip install langchain-openai      # for OpenAI
pip install langchain-anthropic   # for Anthropic Claude
pip install langchain-groq        # for Groq (free tier available)
```

### 4 — Configure environment

```bash
cp .env.example .env
# Edit .env with your keys
```

Minimum required: one LLM API key.  
NCBI API key is optional but recommended (raises rate limit from 3 → 10 req/s).  
Register free at <https://www.ncbi.nlm.nih.gov/account/>.

---

## CLI usage

```bash
# Basic search
python main.py "negative pressure wound therapy diabetic foot ulcers"

# Systematic reviews only, last 3 years, export results
python main.py "pressure ulcer prevention" \
  --article-type systematic_review \
  --date-from 2022/01/01 \
  --max-results 30 \
  --export-dir ./results

# Best dressings for pressure injuries, recent literature
python main.py "best dressings for pressure injuries 2024-2026" \
  --date-from 2024/01/01 \
  --provider anthropic \
  --model claude-opus-4-5

# Interactive REPL
python main.py --interactive

# Groq (fast, free tier)
python main.py "wound biofilm management" --provider groq
```

### All CLI flags

```
positional:
  query                    Natural language search query

optional:
  -n / --max-results N     Max articles to retrieve (default: 20, max: 200)
  -t / --article-type TYPE all | review | systematic_review | meta-analysis |
                           rct | clinical_trial | guideline | case_report
  --date-from YYYY/MM/DD   Earliest publication date
  --date-to   YYYY/MM/DD   Latest publication date
  -o / --export-dir DIR    Export directory for CSV / JSON / BibTeX
  --save-alert             Save search params without prompting
  -p / --provider NAME     openai | anthropic | groq (default: openai)
  -m / --model NAME        Override model (e.g. gpt-4o-mini, claude-haiku-4-5-20251001)
  -v / --verbose           Enable INFO-level logging
  -i / --interactive       Start interactive REPL
```

---

## Streamlit web UI

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.  
Enter API keys in the sidebar or pre-set them in `.env`.

### Google sign-in setup

The app requires Google sign-in (`st.login`) before it can be used. One-time setup:

1. **Google Cloud Console** → [APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials)
   - Configure the **OAuth consent screen** first if you haven't (External, app name, support email — no sensitive scopes needed)
   - Create an **OAuth 2.0 Client ID** → Application type: **Web application**
   - Add **Authorized redirect URIs**:
     - `http://localhost:8501/oauth2callback` (local dev)
     - `https://<your-production-domain>/oauth2callback` (e.g. `https://azwoundcare.info/oauth2callback`)
2. **Local dev** — copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in:
   ```toml
   [auth]
   redirect_uri = "http://localhost:8501/oauth2callback"
   cookie_secret = "<random — see comment in the example file>"
   client_id = "<from Google Cloud Console>"
   client_secret = "<from Google Cloud Console>"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```
   This file is gitignored — never commit it.
3. **Production (Render, or any host without a secrets-file UI)** — set these environment variables, and let `start.sh` generate `.streamlit/secrets.toml` from them at startup:
   - `REDIRECT_URI`, `COOKIE_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`

   Set the Render **Start Command** to:
   ```
   bash start.sh
   ```
   (Don't paste the secrets-generation logic directly into Render's Start Command field — long inline commands with embedded quotes get mangled by the UI. `start.sh` keeps it in a file instead.)

Any Google account can sign in — there is no email allowlist. Add one in `streamlit_app.py` (check `st.user.email` after login) if you need to restrict access later.

---

## Example wound care queries

```text
negative pressure wound therapy diabetic foot ulcers
pressure ulcer prevention spinal cord injury systematic review
best dressings for venous leg ulcers 2022-2025
biofilm management chronic wounds antimicrobial
platelet rich plasma wound healing randomised controlled trial
MRSA wound infection management guidelines
hyperbaric oxygen therapy diabetic foot meta-analysis
debridement methods chronic wound comparison
maggot therapy sloughy wounds evidence
electrical stimulation wound healing systematic review
collagen dressings pressure ulcers elderly
cost effectiveness NPWT surgical wounds
```

---

## Architecture

```
User query
    │
    ▼
refine_wound_care_query   ← maps natural language → MeSH PubMed string
    │
    ▼
search_pubmed             ← ESearch → list of PMIDs + total count
    │
    ▼
fetch_abstracts           ← EFetch (batched 20/req) → article records
    │
    ▼
LLM summarise node        ← structured clinical evidence summary
    │
    ▼
Export (CSV / JSON / BibTeX)
```

LangGraph StateGraph manages state across all nodes. The agent LLM decides
the order of tool calls; the summarise node fires automatically once articles
are available.

---

## Extending to other sources

To add Cochrane, PMC full-text, or Google Scholar:

1. Create a new function in `tools.py` decorated with `@tool(...)`.
2. Append it to the `TOOLS` list in `agent.py`.
3. Update the system prompt in `agent.py` to describe when to use the new tool.

The LangGraph agent will automatically discover and route to the new tool.

---

## Rate limits & NCBI guidelines

| Mode | Rate |
|------|------|
| No API key | 3 requests/second |
| With NCBI API key | 10 requests/second |

The built-in `RateLimiter` enforces 2.5 req/s conservatively. Large batch
fetches (>100 articles) are automatically split into batches of 20 with
inter-batch delays. For bulk downloads register a free NCBI API key.

---

## Saved search alerts

```bash
# Save automatically
python main.py "pressure ulcer" --save-alert

# Re-run all saved alerts (manual loop for now)
python -c "
import json, subprocess
for a in json.load(open('alerts.json')):
    subprocess.run(['python', 'main.py', a['query'], '--export-dir', './results'])
"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `EnvironmentError: OPENAI_API_KEY is not set` | Add key to `.env` or sidebar |
| `HTTPError 429` | Add `NCBI_API_KEY` or reduce `--max-results` |
| Empty results | Broaden query; remove date filter; use `--article-type all` |
| Slow summarisation | Use a faster model: `--model gpt-4o-mini` or Groq |
| XML parse warning | Transient NCBI response issue; usually self-corrects on retry |
