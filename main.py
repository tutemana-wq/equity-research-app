"""
Main script for Equity Research App.

Workflow:
1. Load environment variables from `.env`.
2. Connect to Supabase and read tickers from the `watchlist` table.
3. For each ticker, query SEC-API for 8-K filings from the last 48 hours.
4. For each filing, send the filing text to Gemini 3 Flash via `google-genai`
   to generate a 3-bullet investor risk summary.
5. Use Tavily (or compatible web search) to find recent CEO interviews,
   press releases, YouTube appearances, and investor relations presentations
   for each ticker from the last 72 hours.
6. Summarize all findings into 3-bullet investor risk summaries with Gemini.
7. Save summaries into the `reports` table in Supabase, including a
   'No news found' entry when nothing relevant is detected.

Dependencies (install via pip, e.g. in a virtualenv):
    pip install python-dotenv supabase sec-api google-genai requests resend
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests
import resend
from dotenv import load_dotenv
from sec_api import QueryApi
from supabase import Client, create_client
from google import genai


# ---------- Configuration ----------

WATCHLIST_TABLE = "watchlist"
REPORTS_TABLE = "reports"

# Default Gemini model; can be overridden via GEMINI_MODEL env var
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# Limit text length sent to the LLM to keep prompts manageable
MAX_FILING_CHARS = 20_000

# Lookback window for SEC filings (hours) and web/IR search (hours)
SEC_LOOKBACK_HOURS = 48
WEB_LOOKBACK_HOURS = 72

# Tavily search endpoint (used via HTTP, no SDK required)
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def configure_logging() -> None:
    """Configure basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_environment() -> None:
    """Load environment variables from .env and validate required keys."""
    load_dotenv()

    missing: List[str] = []
    required_keys = [
        "SUPABASE_URL",
        "SUPABASE_KEY",  # service role or anon key with needed perms
        "SEC_API_KEY",
        "GEMINI_API_KEY",  # used by google-genai
        "RESEND_API_KEY",  # used for email sending
    ]

    for key in required_keys:
        if not os.getenv(key):
            missing.append(key)

    if missing:
        raise RuntimeError(
            f"Missing required environment variables in .env: {', '.join(missing)}"
        )


def get_supabase_client() -> Client:
    """Create and return a Supabase client."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def get_gemini_client() -> genai.Client:
    """Create and return a Gemini client using google-genai."""
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    return client


def get_sec_query_api() -> QueryApi:
    """Create and return a SEC-API QueryApi client."""
    api_key = os.environ["SEC_API_KEY"]
    return QueryApi(api_key=api_key)


def fetch_watchlist_tickers(supabase: Client) -> Dict[str, str]:
    """
    Fetch tickers and company names from the watchlist table in Supabase.
    Returns a dict mapping ticker -> company_name (or ticker if company_name not available).
    """
    logging.info("Fetching tickers from Supabase watchlist table '%s'.", WATCHLIST_TABLE)
    try:
        # Try to fetch both ticker and company_name if available
        response = supabase.table(WATCHLIST_TABLE).select("ticker, company_name").execute()
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("Error querying watchlist table: %s", exc, exc_info=True)
        raise

    if not response.data:
        logging.warning("Watchlist table returned no rows.")
        return {}

    ticker_to_name: Dict[str, str] = {}
    for row in response.data:
        ticker = row.get("ticker")
        if isinstance(ticker, str) and ticker.strip():
            ticker_upper = ticker.strip().upper()
            # Use company_name if available, otherwise fall back to ticker
            company_name = row.get("company_name") or ticker_upper
            ticker_to_name[ticker_upper] = company_name

    logging.info("Found %d unique tickers in watchlist.", len(ticker_to_name))
    return ticker_to_name


def query_recent_8k_filings_for_ticker(
    query_api: QueryApi, ticker: str, since: datetime
) -> List[Dict]:
    """
    Use SEC-API's QueryApi to fetch 8-K filings for a ticker since the given datetime.

    SEC-API query syntax example:
      ticker:TSLA AND filedAt:[2020-01-01 TO 2020-12-31] AND formType:"10-Q"
    We'll adapt this to 8-K and limit to the last ~48 hours.
    """
    # Use only date part in query; we'll filter by exact time client-side.
    start_date = since.date().isoformat()
    end_date = datetime.now(timezone.utc).date().isoformat()

    query_string = (
        f"ticker:{ticker} AND filedAt:[{start_date} TO {end_date}] "
        f'AND formType:"8-K"'
    )

    query = {
        "query": query_string,
        "from": "0",
        "size": "50",  # up to 50 recent filings per ticker
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    logging.info("Querying SEC-API for recent 8-K filings for %s.", ticker)
    try:
        result = query_api.get_filings(query)
    except Exception as exc:  # pragma: no cover - network / API issues
        logging.error(
            "Error querying SEC-API for ticker %s: %s", ticker, exc, exc_info=True
        )
        return []

    filings = result.get("filings", []) if isinstance(result, dict) else []

    # Filter by actual datetime (last 48 hours)
    recent_filings: List[Dict] = []
    for filing in filings:
        filed_at_str = filing.get("filedAt")
        if not filed_at_str:
            continue
        try:
            filed_at = datetime.fromisoformat(
                filed_at_str.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        except Exception:
            # If parsing fails, keep the filing (better to include than miss)
            recent_filings.append(filing)
            continue

        if filed_at >= since:
            recent_filings.append(filing)

    return recent_filings


def fetch_filing_text(filing: Dict) -> Optional[str]:
    """
    Download the filing text using the HTML or TXT link returned by SEC-API.

    The QueryApi response typically includes `linkToHtml` and `linkToTxt`.
    We'll try HTML first, then TXT.
    """
    url = filing.get("linkToHtml") or filing.get("linkToTxt")
    if not url:
        logging.warning(
            "No HTML/TXT link found for filing %s; skipping.",
            filing.get("accessionNo", "<unknown>"),
        )
        return None

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network issues
        logging.error("Error downloading filing text from %s: %s", url, exc)
        return None

    text = resp.text
    if not text:
        logging.warning("Downloaded filing text is empty from %s.", url)
        return None

    # Truncate extremely long filings to keep LLM prompt size reasonable
    if len(text) > MAX_FILING_CHARS:
        logging.info(
            "Truncating filing text from %d to %d characters.",
            len(text),
            MAX_FILING_CHARS,
        )
        text = text[:MAX_FILING_CHARS]

    return text


def summarize_filing_with_gemini(
    client: genai.Client, filing_text: str, ticker: str
) -> str:
    """
    Send filing text to Gemini 3 Flash and return a 3-bullet investor risk summary.
    Acts as a gatekeeper - returns "SKIP" for irrelevant content.
    """
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    prompt = (
        f"You are an elite researcher and gatekeeper.\n\n"
        f"CRITICAL: If the provided SEC Form 8-K filing text is a generic market summary, "
        f"a price forecast, routine administrative filing, or not specifically about a major "
        f"company event (like executive changes, material agreements, financial restatements, "
        f"or significant business developments), you MUST return exactly the string \"SKIP\" "
        f"and nothing else. Do not summarize irrelevant noise.\n\n"
        f"If the filing IS relevant and contains material investor risks, summarize it in exactly "
        f"three concise bullet points focused ONLY on investor risks.\n"
        f"- Each bullet should start with '- '.\n"
        f"- Do not include introductions or conclusions.\n\n"
        f"Filing text begins below:\n\n"
        f"{filing_text}"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
    except Exception as exc:  # pragma: no cover - network / API issues
        logging.error("Error calling Gemini for summary: %s", exc, exc_info=True)
        raise

    summary = getattr(response, "text", None) or ""
    return summary.strip()


def search_web_and_ir_for_ticker(ticker: str, company_name: str) -> List[Dict]:
    """
    Use Tavily Search API to find recent CEO interviews, press releases,
    YouTube appearances, and investor relations presentations for the ticker
    within the last WEB_LOOKBACK_HOURS.

    Uses specific intent keywords and excludes generic market summaries,
    price forecasts, and stock price discussions.

    Requires TAVILY_API_KEY in the environment. If it's missing or the API
    fails, this function returns an empty list and logs the issue.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logging.info(
            "TAVILY_API_KEY not set; skipping web and investor relations search "
            "for %s.",
            ticker,
        )
        return []

    now_utc = datetime.now(timezone.utc)
    start_date = (now_utc - timedelta(hours=WEB_LOOKBACK_HOURS)).date().isoformat()
    end_date = now_utc.date().isoformat()

    headers = {"Authorization": f"Bearer {api_key}"}

    base_payload = {
        "search_depth": "advanced",
        "max_results": 10,
        "topic": "finance",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "start_date": start_date,
        "end_date": end_date,
    }

    # Use company name in quotes for exact matching, exclude stock/forecast/price noise
    queries = [
        (
            f'"{company_name}" (interview OR "press release" OR "earnings call") '
            f"-stock -forecast -price -trading -chart -analyst -target"
        ),
        (
            f'"{company_name}" ("investor relations" OR "investor day" OR '
            f'"investor presentation" OR "CEO" OR "executive") '
            f"-stock -forecast -price -trading -chart"
        ),
    ]

    all_results: List[Dict] = []

    for q in queries:
        payload = {**base_payload, "query": q}
        logging.info("Searching Tavily for '%s'.", q)
        try:
            resp = requests.post(
                TAVILY_SEARCH_URL, json=payload, headers=headers, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            all_results.extend(results)
        except Exception as exc:  # pragma: no cover - network / API issues
            logging.error("Error calling Tavily for query '%s': %s", q, exc)
            continue

    # De-duplicate results by URL
    unique_results: List[Dict] = []
    seen_urls = set()
    for r in all_results:
        url = r.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        unique_results.append(r)

    logging.info(
        "Found %d unique web/IR results for %s within the last %d hours.",
        len(unique_results),
        ticker,
        WEB_LOOKBACK_HOURS,
    )
    return unique_results


def summarize_external_with_gemini(
    client: genai.Client, ticker: str, company_name: str, items: List[Dict]
) -> str:
    """
    Summarize external web/IR items (CEO interviews, press releases, YouTube
    appearances, investor presentations) into a 3-bullet investor risk summary.
    Acts as a gatekeeper - returns "SKIP" for irrelevant content.
    """
    if not items:
        return ""

    # Build a combined text from titles, URLs, and content snippets.
    parts: List[str] = []
    for idx, item in enumerate(items, start=1):
        title = item.get("title") or ""
        url = item.get("url") or ""
        content = item.get("content") or ""
        parts.append(
            f"Source {idx}:\nTitle: {title}\nURL: {url}\nKey content: {content}"
        )

    combined_text = "\n\n".join(parts)
    if len(combined_text) > MAX_FILING_CHARS:
        logging.info(
            "Truncating external sources text from %d to %d characters.",
            len(combined_text),
            MAX_FILING_CHARS,
        )
        combined_text = combined_text[:MAX_FILING_CHARS]

    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    prompt = (
        f"You are an elite researcher and gatekeeper.\n\n"
        f"CRITICAL: If the provided text is a generic market summary, a price forecast, "
        f"stock price discussion, analyst target update, trading chart, or not specifically "
        f"about a major company event (like CEO interviews, press releases about business "
        f"developments, earnings calls, investor presentations, or significant corporate "
        f"announcements), you MUST return exactly the string \"SKIP\" and nothing else. "
        f"Do not summarize irrelevant noise.\n\n"
        f"If the content IS relevant and contains material investor risks or significant "
        f"company developments for {company_name} ({ticker}), summarize it in exactly "
        f"three concise bullet points focused ONLY on investor risks.\n"
        f"- Each bullet should start with '- '.\n"
        f"- Do not include introductions or conclusions.\n\n"
        f"Sources:\n\n"
        f"{combined_text}"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
    except Exception as exc:  # pragma: no cover - network / API issues
        logging.error(
            "Error calling Gemini for external news summary: %s", exc, exc_info=True
        )
        raise

    summary = getattr(response, "text", None) or ""
    return summary.strip()


def _format_summary_text(
    raw_summary: str,
    headline: Optional[str],
    source_url: Optional[str],
    is_no_news: bool = False,
) -> str:
    """
    Format the final summary text:
    - For news: Headline (hyperlinked) + bullet points
    - For no news: Simple "No news." message
    """
    if is_no_news:
        # Simple format for "No news" entries
        return "No news."
    
    # Normalize and enforce bullet formatting.
    lines = [line.strip() for line in raw_summary.splitlines() if line.strip()]
    bullets: List[str] = []
    for line in lines:
        if line.startswith("-"):
            bullets.append(line)
        else:
            bullets.append(f"- {line}")

    # Ensure we don't exceed 3 bullets; if Gemini returned more, keep only the first 3.
    if len(bullets) > 3:
        bullets = bullets[:3]

    bullets_block = "\n".join(bullets)

    # Build headline with clickable link using markdown syntax.
    if headline and source_url:
        headline_text = f"[{headline}]({source_url})"
    elif source_url:
        headline_text = f"[{source_url}]({source_url})"
    else:
        headline_text = headline or "N/A"

    formatted = (
        f"The Headline: {headline_text}\n\n"
        f"{bullets_block}"
    )
    return formatted


def save_summary_to_supabase(
    supabase: Client,
    ticker: str,
    filing: Dict,
    summary: str,
    company_name: Optional[str] = None,
    source_url: Optional[str] = None,
    headline: Optional[str] = None,
) -> None:
    """Insert the formatted summary for a filing into the Supabase reports table."""
    if not summary:
        logging.warning(
            "Empty summary for ticker %s; skipping insert into reports table.", ticker
        )
        return

    filed_at = filing.get("filedAt")
    accession_no = filing.get("accessionNo")
    form_type = filing.get("formType")

    filing_url = filing.get("linkToHtml") or filing.get("linkToTxt")
    if source_url is None:
        source_url = filing_url

    formatted_summary = _format_summary_text(summary, headline, source_url)

    payload = {
        "ticker": ticker,
        "company_name": company_name or ticker,  # Store company_name in database
        "summary": formatted_summary,
        "filing_date": filed_at,
        "filing_type": form_type,
        "accession_no": accession_no,
        "filing_url": filing_url,
        "source_url": source_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    logging.info(
        "Inserting summary into Supabase reports table '%s' for %s (%s).",
        REPORTS_TABLE,
        ticker,
        accession_no,
    )
    try:
        supabase.table(REPORTS_TABLE).insert(payload).execute()
    except Exception as exc:  # pragma: no cover - DB / network issues
        logging.error("Error inserting report into Supabase: %s", exc, exc_info=True)


def process_ticker(
    ticker: str,
    company_name: str,
    query_api: QueryApi,
    supabase: Client,
    gemini_client: genai.Client,
) -> List[Dict]:
    """
    Process a single ticker: query filings, summarize, and store results.
    Returns a list of summary dicts for email compilation.
    """
    logging.info("Processing ticker %s (%s).", ticker, company_name)
    any_summary_saved = False
    email_summaries: List[Dict] = []

    # --- SEC 8-K filings (last 48 hours) ---
    since_sec = datetime.now(timezone.utc) - timedelta(hours=SEC_LOOKBACK_HOURS)
    filings = query_recent_8k_filings_for_ticker(query_api, ticker, since_sec)

    if not filings:
        logging.info(
            "No recent 8-K filings found for %s in the last %d hours.",
            ticker,
            SEC_LOOKBACK_HOURS,
        )
    else:
        for filing in filings:
            accession_no = filing.get("accessionNo", "<unknown>")
            logging.info("Handling filing %s for %s.", accession_no, ticker)

            try:
                filing_text = fetch_filing_text(filing)
                if not filing_text:
                    logging.info(
                        "Skipping filing %s due to missing text.", accession_no
                    )
                    continue

                summary = summarize_filing_with_gemini(
                    gemini_client, filing_text, ticker
                )
                
                # Check if Gemini returned "SKIP" - treat as no news
                if summary.strip().upper() == "SKIP":
                    logging.info(
                        "Gemini gatekeeper returned SKIP for filing %s (%s); treating as irrelevant.",
                        accession_no,
                        ticker,
                    )
                    continue

                filing_url = filing.get("linkToHtml") or filing.get("linkToTxt")
                filing_title = f"SEC Form {filing.get('formType', '8-K')} - {company_name} ({ticker})"
                
                save_summary_to_supabase(
                    supabase, ticker, filing, summary, company_name=company_name, source_url=filing_url, headline=filing_title
                )
                any_summary_saved = True
                
                # Collect for email
                formatted_summary = _format_summary_text(summary, filing_title, filing_url, is_no_news=False)
                email_summaries.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "summary": formatted_summary,
                    "source_url": filing_url,
                    "is_no_news": False,
                })
            except Exception as exc:  # pragma: no cover - per-filing robustness
                logging.error(
                    "Error processing filing %s for %s: %s",
                    accession_no,
                    ticker,
                    exc,
                    exc_info=True,
                )
                # Continue with the next filing instead of crashing
                continue

    # --- Web search & Investor Relations (last 72 hours) ---
    try:
        external_items = search_web_and_ir_for_ticker(ticker, company_name)
    except Exception as exc:  # pragma: no cover - per-ticker robustness
        logging.error(
            "Unexpected error while searching web/IR for %s: %s",
            ticker,
            exc,
            exc_info=True,
        )
        external_items = []

    if external_items:
        try:
            external_summary = summarize_external_with_gemini(
                gemini_client, ticker, company_name, external_items
            )
            
            # Check if Gemini returned "SKIP" - treat as no news
            if external_summary.strip().upper() == "SKIP":
                logging.info(
                    "Gemini gatekeeper returned SKIP for external news (%s); treating as irrelevant.",
                    ticker,
                )
                # Don't set any_summary_saved = True, so "No news" entry will be created
            else:
                # Use the top-ranked external result as the primary source for the headline.
                primary = external_items[0]
                primary_url = primary.get("url")
                primary_title = primary.get("title") or primary_url

                # Fake a filing-like object to reuse the same Supabase insert helper.
                fake_filing = {
                    "filedAt": datetime.now(timezone.utc).isoformat(),
                    "formType": "NEWS_IR",
                    "accessionNo": None,
                    "linkToHtml": primary_url,
                }
                save_summary_to_supabase(
                    supabase,
                    ticker,
                    fake_filing,
                    external_summary,
                    company_name=company_name,
                    source_url=primary_url,
                    headline=primary_title,
                )
                any_summary_saved = True
                
                # Collect for email
                formatted_summary = _format_summary_text(external_summary, primary_title, primary_url, is_no_news=False)
                email_summaries.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "summary": formatted_summary,
                    "source_url": primary_url,
                    "is_no_news": False,
                })
        except Exception as exc:  # pragma: no cover - per-ticker robustness
            logging.error(
                "Error summarizing external news/IR for %s: %s",
                ticker,
                exc,
                exc_info=True,
            )

    # --- If absolutely no news found, still create a 'No news found' report ---
    if not any_summary_saved:
        logging.info(
            "No SEC filings or external news/IR found for %s (%s) in the configured "
            "lookback windows; creating 'No news found' report entry.",
            ticker,
            company_name,
        )
        no_news_filing = {
            "filedAt": datetime.now(timezone.utc).isoformat(),
            "formType": "NONE",
            "accessionNo": None,
            "linkToHtml": None,
        }
        no_news_summary = f"{company_name}: No news"
        
        # Simple format for "No news" entries
        formatted_no_news = _format_summary_text("", company_name, None, is_no_news=True)
        
        save_summary_to_supabase(supabase, ticker, no_news_filing, no_news_summary, company_name=company_name)
        
        # Collect for email
        email_summaries.append({
            "ticker": ticker,
            "company_name": company_name,
            "summary": formatted_no_news,
            "source_url": None,
            "is_no_news": True,
        })
    
    return email_summaries


def format_summaries_as_html(all_summaries: List[Dict]) -> str:
    """Format all summaries into HTML email body."""
    if not all_summaries:
        return "<p>No summaries to report.</p>"
    
    html_parts = ["<html><body style='font-family: Arial, sans-serif; line-height: 1.6;'>"]
    
    for item in all_summaries:
        ticker = item["ticker"]
        company_name = item["company_name"]
        summary_text = item["summary"]
        source_url = item.get("source_url")
        is_no_news = item.get("is_no_news", False)
        
        html_parts.append(f"<div style='margin-bottom: 30px; padding: 15px; border-left: 4px solid #007bff;'>")
        html_parts.append(f"<h2 style='color: #333; margin-top: 0;'>{company_name} ({ticker})</h2>")
        
        # Handle "No news" case - simple text
        if is_no_news or summary_text.strip() == "No news.":
            html_parts.append("<p style='color: #666; font-style: italic;'>No news.</p>")
        else:
            # Parse the formatted summary text for news items
            lines = summary_text.split("\n")
            headline_processed = False
            bullets = []
            
            for line in lines:
                line = line.strip()
                if line.startswith("The Headline:"):
                    headline_text = line.replace("The Headline:", "").strip()
                    # Parse markdown link format [text](url) if present
                    markdown_link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
                    match = re.search(markdown_link_pattern, headline_text)
                    if match:
                        link_text = match.group(1)
                        link_url = match.group(2)
                        # News title as hyperlinked subheadline
                        html_parts.append(f"<h3 style='color: #007bff; margin-top: 10px;'><a href='{link_url}' style='color: #007bff; text-decoration: none;'>{link_text}</a></h3>")
                    elif source_url and source_url != "N/A":
                        # Fallback to source_url if no markdown link found
                        html_parts.append(f"<h3 style='color: #007bff; margin-top: 10px;'><a href='{source_url}' style='color: #007bff; text-decoration: none;'>{headline_text}</a></h3>")
                    else:
                        html_parts.append(f"<h3 style='color: #007bff; margin-top: 10px;'>{headline_text}</h3>")
                    headline_processed = True
                elif line and not line.startswith("The Headline:"):
                    # Extract bullet content (remove leading "- ")
                    bullet_content = line.lstrip("- ").strip()
                    if bullet_content:
                        bullets.append(bullet_content)
            
            # Add bullets as a list (without "The Body:" label)
            if bullets:
                html_parts.append("<ul style='margin-left: 20px; margin-top: 10px;'>")
                for bullet in bullets:
                    html_parts.append(f"<li style='margin-bottom: 5px;'>{bullet}</li>")
                html_parts.append("</ul>")
        
        html_parts.append("</div>")
    
    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def send_email_via_resend(
    html_body: str,
    subject: str,
    recipient_email: str,
) -> None:
    """Send email using Resend API."""
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("ERROR: RESEND_API_KEY not found; cannot send email.")
        logging.error("RESEND_API_KEY not found; cannot send email.")
        return
    
    try:
        print("Attempting to send email...")
        print(f"Recipient: {recipient_email}")
        print(f"Subject: {subject}")
        
        # Initialize Resend API key
        resend.api_key = api_key
        
        # Use onboarding@resend.dev as the sender (Resend's default verified domain)
        sender_email = "onboarding@resend.dev"
        print(f"Sender: {sender_email}")
        
        params = {
            "from": sender_email,
            "to": recipient_email,
            "subject": subject,
            "html": html_body,
        }
        
        print(f"Calling resend.Emails.send() with params: {list(params.keys())}")
        print(f'Sending email to {recipient_email}...')
        result = resend.Emails.send(params)
        print(f'Resend Response: {result}')
        if result:
            success_id = result.get('id')
            print(f"SUCCESS: Email sent successfully to {recipient_email}")
            print(f"Success ID: {success_id}")
            logging.info(f"Email sent successfully to {recipient_email}. Message ID: {success_id}")
        else:
            print("WARNING: Resend returned None or empty response")
            logging.warning("Resend returned None or empty response")
    except Exception as exc:
        print(f"ERROR: Failed to send email via Resend:")
        print(f"  Exception type: {type(exc).__name__}")
        print(f"  Exception message: {str(exc)}")
        print(f"  Full exception details:")
        import traceback
        traceback.print_exc()
        logging.error(f"Error sending email via Resend: {exc}", exc_info=True)


def main() -> None:
    """Entry point for the script."""
    configure_logging()

    try:
        load_environment()
    except RuntimeError as exc:
        logging.critical(str(exc))
        return

    supabase = get_supabase_client()
    gemini_client = get_gemini_client()
    query_api = get_sec_query_api()

    try:
        ticker_to_name = fetch_watchlist_tickers(supabase)
    except Exception:
        logging.critical("Failed to fetch tickers from watchlist. Exiting.")
        return

    if not ticker_to_name:
        logging.info("No tickers to process. Exiting.")
        return

    # Collect all summaries for email
    all_summaries: List[Dict] = []

    # Process tickers in parallel using ThreadPoolExecutor for speed optimization
    print(f"\n{'='*60}")
    print(f"Processing {len(ticker_to_name)} tickers in parallel...")
    print(f"{'='*60}")
    
    def process_ticker_wrapper(args):
        """Wrapper function for parallel processing."""
        ticker, company_name = args
        try:
            return process_ticker(ticker, company_name, query_api, supabase, gemini_client)
        except Exception as exc:
            logging.error(
                "Unexpected error while processing ticker %s: %s",
                ticker,
                exc,
                exc_info=True,
            )
            # Return empty list on error, but ensure "No news" entry is still created
            # Create a "No news" entry for this ticker even if processing failed
            return [{
                "ticker": ticker,
                "company_name": company_name,
                "summary": (
                    f"The Headline: {company_name}\n\n"
                    f"The Body:\n"
                    f"- Error processing ticker: {str(exc)}\n\n"
                    f"The Data:\n"
                    f"source_url: N/A"
                ),
                "source_url": None,
            }]
    
    # Use ThreadPoolExecutor to process all tickers in parallel
    with ThreadPoolExecutor(max_workers=min(len(ticker_to_name), 8)) as executor:
        # Submit all ticker processing tasks
        future_to_ticker = {
            executor.submit(process_ticker_wrapper, (ticker, company_name)): (ticker, company_name)
            for ticker, company_name in ticker_to_name.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_ticker):
            ticker, company_name = future_to_ticker[future]
            try:
                summaries = future.result()
                all_summaries.extend(summaries)
                print(f"✓ Completed: {company_name} ({ticker}) - {len(summaries)} summary(ies)")
            except Exception as exc:
                logging.error(f"Error getting result for {ticker}: {exc}", exc_info=True)
                # Ensure "No news" entry even if there was an error
                all_summaries.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "summary": "No news.",
                    "source_url": None,
                    "is_no_news": True,
                })
            print("Rate limit protection: Sleeping for 10 seconds before the next ticker...")
            time.sleep(10)

    # TICKER PROCESSING FINISHED - Now sending email
    print(f"\n{'='*60}")
    print(f"PARALLEL PROCESSING COMPLETED. Processed {len(ticker_to_name)} tickers.")
    print(f"Total summaries collected: {len(all_summaries)}")
    print(f"{'='*60}")
    
    # Send email with all summaries (OUTSIDE the ticker loop - single master email)
    print(f"\n{'='*60}")
    print(f"=== Email Preparation (AFTER TICKER LOOP) ===")
    print(f"{'='*60}")
    print(f"Total summaries collected: {len(all_summaries)}")
    
    today_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
    subject = f"Daily Micro-Cap Intelligence Briefing - {today_str}"
    
    html_body = format_summaries_as_html(all_summaries)
    print(f"HTML body length: {len(html_body)} characters")
    
    # Get recipient email from environment - REQUIRED
    recipient_email = os.getenv("EMAIL_RECIPIENT")
    print(f"EMAIL_RECIPIENT from .env: {recipient_email}")
    
    if not recipient_email:
        print("ERROR: EMAIL_RECIPIENT not set in .env; cannot send email.")
        logging.error("EMAIL_RECIPIENT not set in .env; cannot send email.")
        logging.info("Email body would have been:\n%s", html_body[:500])
        return
    
    # Ensure we have summaries for ALL tickers (including "No news" entries)
    expected_count = len(ticker_to_name)
    if len(all_summaries) < expected_count:
        print(f"WARNING: Expected {expected_count} summaries (one per ticker), but got {len(all_summaries)}")
        logging.warning(f"Expected {expected_count} summaries, but got {len(all_summaries)}")
        # Find missing tickers and add "No news" entries
        tickers_with_summaries = {s["ticker"] for s in all_summaries}
        for ticker, company_name in ticker_to_name.items():
            if ticker not in tickers_with_summaries:
                print(f"Adding missing 'No news' entry for {company_name} ({ticker})")
                all_summaries.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "summary": "No news.",
                    "source_url": None,
                    "is_no_news": True,
                })
        # Re-format HTML body with complete summaries
        html_body = format_summaries_as_html(all_summaries)
    
    # Send email at the very end
    print(f"\n{'='*60}")
    print(f"=== CALLING send_email_via_resend() AT THE END ===")
    print(f"{'='*60}")
    send_email_via_resend(html_body, subject, recipient_email)
    print(f"{'='*60}")
    print("=== Email Process Complete ===")
    print(f"{'='*60}\n")

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
