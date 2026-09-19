# BOB Reputation Risk Analyzer — Streamlit Dashboard

A media-sentiment / reputation-risk dashboard for tracking news coverage on
one or more companies. This is a Streamlit rewrite of the original Flask +
ngrok / Colab notebook version — same scraping and scoring logic, but as a
single deployable app with everything displayed upfront and one-click Excel
downloads.

## What it does

1. You enter one or more companies, a date range per company, and (optionally)
   custom sentiment words/scores.
2. For each company, it:
   - Searches for recent Moneycontrol news coverage (3-tier fallback: direct
     Google scrape → `googlesearch-python` → DuckDuckGo).
   - Scrapes and de-duplicates the matching articles in the date range.
   - Scores every article three ways:
     - **NPS** — a lexicon-based net percentage score (positive vs. negative
       word counts, VADER lexicon + your custom words).
     - **NLPS** — NLTK VADER compound sentiment score.
     - **LLM** — a local transformer model
       (`cardiffnlp/twitter-roberta-base-sentiment-latest`), no API key needed.
3. Renders KPI cards (color-coded green ≥ 0.5 / amber ≥ 0.25 / red < 0.25),
   an interactive sentiment time-series chart, a word intensity/frequency
   bubble chart, and the full article table — all upfront, no polling.
4. Lets you download a formatted Excel report per company (Data + Dashboard
   sheet with embedded charts, matching the original layout) or all of them
   at once as a ZIP.

## ⚠️ Before you push this to GitHub

The original notebook (`BOB_Reputation_Risk_Analyzer.ipynb`) has a **live
ngrok auth token and a Gemini API key hardcoded in cell 2**. If that notebook
(or any earlier commit containing it) goes into this repo:

- **Rotate/revoke both keys immediately** — treat them as compromised.
- Do not commit the `.ipynb` file as-is. If you want to keep it for
  reference, strip the secrets first or keep it out of version control
  (see `.gitignore`).

This Streamlit app itself needs **no API keys** — the "LLM" score comes from
a local HuggingFace model, not a hosted API.

## Project structure

```
.
├── app.py              # the Streamlit app
├── requirements.txt
├── README.md
└── .gitignore
```

## Running locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the VADER lexicon and the RoBERTa sentiment model
(~500 MB) — this is cached after that (`st.cache_resource`), so subsequent
runs and reruns are fast.

## Deploying (Streamlit Community Cloud)

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at
   `app.py`.
3. No secrets to configure. Note that `transformers` + `torch` + the model
   weights are sizeable — Streamlit Cloud's free tier (1 GB RAM) can be tight.
   If you hit memory/build issues:
   - Keep the pinned CPU-only `torch` wheel in `requirements.txt` (already set
     via `--extra-index-url https://download.pytorch.org/whl/cpu`).
   - Consider swapping the RoBERTa model for a smaller one, or moving to a
     host with more RAM (Render, an internal VM, etc.) if usage grows.

## Notes & caveats

- **Scraping**: this pulls public pages from moneycontrol.com. Keep request
  volume reasonable and review moneycontrol's terms of use for your
  organization's intended usage — the app already rate-limits itself lightly
  between requests, matching the original notebook's behavior.
- **Google search scraping** (tier 1 fallback) can get rate-limited/blocked
  by Google in cloud environments; the app automatically falls back to
  `googlesearch-python` and then DuckDuckGo.
- **Brand colors** used (`#F7931E` orange / `#6E1E33` maroon) are close
  approximations of Bank of Baroda's palette. Swap the `BOB_ORANGE` /
  `BOB_MAROON` constants at the top of `app.py` for exact brand hex codes if
  you have official guidelines.
