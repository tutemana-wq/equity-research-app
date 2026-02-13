"""
Universal SEC & News Monitor - Automated Daily Briefing

Workflow:
1. Load environment variables from `.env`.
2. Connect to Supabase and read tickers from the `watchlist` table.
3. For each ticker, query SEC EDGAR (free public API) for ALL filing types from the last 24 hours.
   - Categorize filings: Crucial (8-K, 10-Q, 10-K), Insider/Ownership (Form 3/4/5, SC 13D/G), Other
4. For each filing, send the filing text to Gemini 3 Flash (with Anthropic failover)
   to generate a 3-bullet investor risk summary.
5. Use Tavily advanced search to find official investor relations news and press releases
   for each ticker from the last 72 hours.
   - Prioritize URLs from: businesswire, globenewswire, prnewswire, investor
6. Summarize all findings into 3-bullet investor risk summaries with AI (Gemini → Anthropic failover).
7. Save summaries into the `reports` table in Supabase, including a
   'No news found' entry when nothing relevant is detected.
8. Send email with all findings, each with direct clickable links.

Dependencies (install via pip, e.g. in a virtualenv):
    pip install python-dotenv supabase google-genai requests resend anthropic
"""

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests
import resend
from dotenv import load_dotenv
from supabase import Client, create_client
from google import genai

# Try to import Anthropic for failover
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


# ---------- Configuration ----------

WATCHLIST_TABLE = "watchlist"
REPORTS_TABLE = "reports"

# Default Gemini model; can be overridden via GEMINI_MODEL env var
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# Limit text length sent to the LLM to keep prompts manageable
MAX_FILING_CHARS = 20_000

# Lookback window for SEC filings (hours) and web/IR search (hours)
# SEC: 48 hours normally, 72 hours on Monday (to catch Friday filings)
SEC_LOOKBACK_HOURS = 48
WEB_LOOKBACK_HOURS = 72

# Filing categorization
CRUCIAL_FILINGS = ["8-K", "10-Q", "10-K", "10-Q/A", "10-K/A", "8-K/A"]
INSIDER_OWNERSHIP_FILINGS = ["3", "4", "5", "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"]

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
        "SEC_USER_AGENT",  # replaces SEC_API_KEY
        "GEMINI_API_KEY",  # used by google-genai (primary AI)
        "RESEND_API_KEY",  # used for email sending
    ]

    optional_keys = [
        "ANTHROPIC_API_KEY",  # used for AI failover when Gemini quota exceeded
    ]

    for key in required_keys:
        if not os.getenv(key):
            missing.append(key)

    if missing:
        raise RuntimeError(
            f"Missing required environment variables in .env: {', '.join(missing)}"
        )

    # Check optional keys and log warnings
    for key in optional_keys:
        if not os.getenv(key):
            logging.info("Optional environment variable '%s' not set. AI failover will not be available.", key)

    # Validate SEC_USER_AGENT format
    sec_user_agent = os.getenv("SEC_USER_AGENT", "")
    if sec_user_agent and not re.match(r".+\s+<.+@.+\..+>", sec_user_agent):
        logging.warning(
            "SEC_USER_AGENT should be in format 'Name <email@example.com>' "
            "to comply with SEC fair access policy."
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


# Global cache for company tickers mapping
_COMPANY_TICKERS_CACHE: Optional[Dict[str, int]] = None


def get_company_tickers_mapping() -> Dict[str, int]:
    """
    Download and cache SEC company tickers to CIK mapping.
    Returns a dict mapping uppercase ticker -> CIK integer.
    Uses session-level caching to avoid repeated downloads.
    """
    global _COMPANY_TICKERS_CACHE

    if _COMPANY_TICKERS_CACHE is not None:
        return _COMPANY_TICKERS_CACHE

    url = "https://www.sec.gov/files/company_tickers.json"
    user_agent = os.getenv("SEC_USER_AGENT", "Unknown <unknown@unknown.com>")
    headers = {"User-Agent": user_agent}

    logging.info("Downloading SEC company tickers mapping from %s", url)
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logging.error("Error downloading company tickers mapping: %s", exc, exc_info=True)
        return {}

    # Parse JSON structure: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
    mapping: Dict[str, int] = {}
    for entry in data.values():
        if isinstance(entry, dict):
            ticker = entry.get("ticker")
            cik_str = entry.get("cik_str")
            if ticker and cik_str is not None:
                mapping[ticker.upper()] = int(cik_str)

    _COMPANY_TICKERS_CACHE = mapping
    return mapping


def ticker_to_cik(ticker: str, mapping: Dict[str, int]) -> Optional[int]:
    """
    Look up CIK number for a ticker using the provided mapping.
    Returns CIK integer or None if not found.
    """
    return mapping.get(ticker.upper())


def fetch_company_submissions(cik: int) -> Dict:
    """
    Fetch company submissions data from SEC EDGAR API.
    CIK is zero-padded to 10 digits for the API request.
    Returns full submissions JSON or empty dict on error.
    Includes rate limiting delay (0.11s) to stay under SEC's 10 req/sec limit.
    """
    # Zero-pad CIK to 10 digits for SEC API
    cik_padded = f"{cik:010d}"
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    user_agent = os.getenv("SEC_USER_AGENT", "Unknown <unknown@unknown.com>")
    headers = {"User-Agent": user_agent}

    logging.debug("Fetching submissions for CIK %s from %s", cik_padded, url)

    # Rate limiting: 0.11 seconds = ~9 requests/second (SEC allows 10)
    time.sleep(0.11)

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logging.error("Error fetching submissions for CIK %s: %s", cik, exc, exc_info=True)
        return {}


def construct_filing_urls(cik: int, accession_number: str, primary_document: str) -> Dict[str, str]:
    """
    Construct HTML and TXT URLs for a SEC filing.

    Args:
        cik: Company CIK (will be used without padding in URL)
        accession_number: Accession number with dashes (e.g., "0001234567-23-000001")
        primary_document: Primary document filename (e.g., "aapl-20230701.htm")

    Returns:
        Dict with "linkToHtml" and "linkToTxt" keys
    """
    # Remove dashes from accession number for URL path
    acc_no_nodashes = accession_number.replace("-", "")

    # Build URLs (CIK without padding)
    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_nodashes}"

    urls = {
        "linkToTxt": f"{base_url}.txt"
    }

    # Add HTML URL only if primary document is provided
    if primary_document:
        urls["linkToHtml"] = f"{base_url}/{primary_document}"

    return urls


def categorize_filing(form_type: str) -> str:
    """
    Categorize a filing type into Crucial, Insider/Ownership, or Other.

    Returns:
        "Crucial", "Insider/Ownership", or "Other"
    """
    if form_type in CRUCIAL_FILINGS:
        return "Crucial"
    elif form_type in INSIDER_OWNERSHIP_FILINGS:
        return "Insider/Ownership"
    else:
        return "Other"


def parse_recent_filings(submissions_data: Dict, form_type: Optional[str], since: datetime) -> List[Dict]:
    """
    Parse recent filings from submissions data.
    Filters by form type (optional) and filing date.

    Args:
        submissions_data: Full submissions JSON from SEC API
        form_type: Form type to filter (e.g., "8-K"), or None to fetch all types
        since: Only include filings accepted on or after this datetime

    Returns:
        List of filing dicts with keys: accessionNo, filedAt, formType, primaryDocument, category
    """
    filings_section = submissions_data.get("filings", {}).get("recent", {})
    if not filings_section:
        return []

    # SEC data is columnar - same index across all arrays represents one filing
    accession_numbers = filings_section.get("accessionNumber", [])
    filing_dates = filings_section.get("filingDate", [])
    acceptance_datetimes = filings_section.get("acceptanceDateTime", [])
    forms = filings_section.get("form", [])
    primary_documents = filings_section.get("primaryDocument", [])

    result: List[Dict] = []

    for i in range(len(accession_numbers)):
        # Filter by form type (exact match) - skip if form_type specified and doesn't match
        current_form = forms[i] if i < len(forms) else None
        if form_type is not None and current_form != form_type:
            continue

        # Filter by acceptance datetime
        if i < len(acceptance_datetimes):
            acceptance_dt_str = acceptance_datetimes[i]
            try:
                acceptance_dt = datetime.fromisoformat(
                    acceptance_dt_str.replace("Z", "+00:00")
                ).astimezone(timezone.utc)

                if acceptance_dt < since:
                    continue
            except Exception as exc:
                logging.warning("Error parsing acceptance datetime '%s': %s", acceptance_dt_str, exc)
                # Include the filing if we can't parse the date (better to include than miss)

        # Build filing dict with category
        filing_form_type = forms[i] if i < len(forms) else "Unknown"
        filing = {
            "accessionNo": accession_numbers[i] if i < len(accession_numbers) else None,
            "filedAt": acceptance_datetimes[i] if i < len(acceptance_datetimes) else (
                filing_dates[i] if i < len(filing_dates) else None
            ),
            "formType": filing_form_type,
            "primaryDocument": primary_documents[i] if i < len(primary_documents) else None,
            "category": categorize_filing(filing_form_type),
        }
        result.append(filing)

    return result


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


def query_recent_filings_for_ticker(
    ticker: str, since: datetime, ticker_cik_mapping: Dict[str, int], form_type: Optional[str] = None
) -> List[Dict]:
    """
    Fetch all SEC filings for a ticker from SEC EDGAR using free public APIs.
    Uses direct SEC access, replacing the paid sec-api service.

    Args:
        ticker: Stock ticker symbol
        since: Only return filings accepted on or after this datetime
        ticker_cik_mapping: Dict mapping ticker -> CIK number
        form_type: Optional form type to filter (e.g., "8-K"), None for all types

    Returns:
        List of filing dicts with category, formType, URLs, etc.
    """
    filing_msg = f"all filings" if form_type is None else f"{form_type} filings"
    logging.info("Querying SEC EDGAR for recent %s for %s.", filing_msg, ticker)

    # Look up CIK for ticker
    cik = ticker_to_cik(ticker, ticker_cik_mapping)
    if cik is None:
        logging.warning("Ticker %s not found in SEC company tickers mapping.", ticker)
        return []

    # Fetch company submissions from SEC
    submissions = fetch_company_submissions(cik)
    if not submissions:
        logging.warning("No submissions data found for ticker %s (CIK: %s).", ticker, cik)
        return []

    # Parse recent filings (all types if form_type is None)
    filings = parse_recent_filings(submissions, form_type, since)

    # Add URLs to each filing
    for filing in filings:
        accession_no = filing.get("accessionNo")
        primary_doc = filing.get("primaryDocument")

        if accession_no:
            urls = construct_filing_urls(cik, accession_no, primary_doc)
            filing.update(urls)

    logging.info("Found %d recent %s for %s.", len(filings), filing_msg, ticker)
    return filings


def fetch_filing_text(filing: Dict) -> Optional[str]:
    """
    Download the filing text using the HTML or TXT link.

    The filing dict includes `linkToHtml` and `linkToTxt`.
    We'll try HTML first, then TXT.
    """
    url = filing.get("linkToHtml") or filing.get("linkToTxt")
    if not url:
        logging.warning(
            "No HTML/TXT link found for filing %s; skipping.",
            filing.get("accessionNo", "<unknown>"),
        )
        return None

    # Add User-Agent header for SEC compliance
    user_agent = os.getenv("SEC_USER_AGENT", "Unknown <unknown@unknown.com>")
    headers = {"User-Agent": user_agent}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
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


def generate_content_with_failover(prompt: str, gemini_client: genai.Client) -> str:
    """
    Generate content using Gemini first, with automatic failover to Anthropic if quota exceeded.

    Args:
        prompt: The prompt to send to the AI
        gemini_client: Gemini client instance

    Returns:
        Generated text response
    """
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    # Try Gemini first (free)
    try:
        logging.debug("Attempting to generate content with Gemini...")
        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt,
        )
        summary = getattr(response, "text", None) or ""
        return summary.strip()
    except Exception as exc:
        error_str = str(exc)

        # Check if it's a quota/rate limit error
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            logging.warning("Gemini quota exceeded, attempting failover to Anthropic...")

            # Failover to Anthropic (paid)
            if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
                try:
                    anthropic_client = anthropic.Anthropic(
                        api_key=os.environ["ANTHROPIC_API_KEY"]
                    )

                    message = anthropic_client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1024,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )

                    response_text = message.content[0].text if message.content else ""
                    logging.info("Successfully used Anthropic failover.")
                    return response_text.strip()
                except Exception as anthropic_exc:
                    logging.error("Anthropic failover also failed: %s", anthropic_exc, exc_info=True)
                    raise
            else:
                logging.error("Anthropic not available for failover (missing library or API key)")
                raise

        # If not a quota error, re-raise
        logging.error("Error calling Gemini (non-quota): %s", exc, exc_info=True)
        raise


def summarize_filing_with_gemini(
    client: genai.Client, filing_text: str, ticker: str
) -> str:
    """
    Send filing text to Gemini 3 Flash and return a 3-bullet investor risk summary.
    Acts as a gatekeeper - returns "SKIP" for irrelevant content.
    """
    print("--- Rate Limit Protection: Sleeping for 12 seconds ---")
    time.sleep(12)

    prompt = (
        f"You are an elite researcher and gatekeeper.\n\n"
        f"CRITICAL: If the provided SEC filing text is a generic market summary, "
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

    return generate_content_with_failover(prompt, client)


def search_web_and_ir_for_ticker(ticker, company_name):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key: return []
clean_name = company_name.replace(", Inc.", "").replace(" Inc.", "").replace(" Corp.", "").replace(" Ltd.", "").strip()
    now_utc = datetime.now(timezone.utc)
    current_year = now_utc.year
headers = {"Authorization": f"Bearer {api_key}"}
    base_payload = {
        "search_depth": "advanced",
        "max_results": 5,
        "topic": "news", 
        "include_answer": False,
        "start_date": (now_utc - timedelta(hours=WEB_LOOKBACK_HOURS)).date().isoformat(),
        "end_date": now_utc.date().isoformat(),
    }
 queries = [
        f"{clean_name} {ticker} investor relations news {current_year}",
        f"{clean_name} {ticker} earnings press release {current_year}",
        f"{clean_name} {ticker} official announcement {current_year}"
    ]
    all_results = []
    for q in queries:
        try:
            resp = requests.post("https://api.tavily.com/search", json={**base_payload, "query": q}, headers=headers, timeout=30)
            if resp.status_code == 200:
                all_results.extend(resp.json().get("results", []))
        except Exception as exc:
            logging.error(f"Search failed for {q}: {exc}")
unique_results = []
    seen_urls = set()
    for r in all_results:
        url = r.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
logging.info("Found %d unique results for %s.", len(unique_results), ticker)
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

    print("--- Rate Limit Protection: Sleeping for 12 seconds ---")
    time.sleep(12)

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

    return generate_content_with_failover(prompt, client)


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
    ticker_cik_mapping: Dict[str, int],
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

    # --- All SEC filings (last 48 hours, or 72 hours on Monday) ---
    now_utc = datetime.now(timezone.utc)

    # If today is Monday (weekday 0), look back 72 hours to catch Friday filings
    if now_utc.weekday() == 0:  # Monday
        lookback_hours = 72
        logging.info("Monday detected: extending SEC lookback window to 72 hours to catch Friday filings.")
    else:
        lookback_hours = SEC_LOOKBACK_HOURS

    since_sec = now_utc - timedelta(hours=lookback_hours)
    filings = query_recent_filings_for_ticker(ticker, since_sec, ticker_cik_mapping, form_type=None)

    if not filings:
        logging.info(
            "No recent SEC filings found for %s in the last %d hours.",
            ticker,
            lookback_hours,
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
                form_type = filing.get('formType', 'Unknown')
                category = filing.get('category', 'Other')
                filing_title = f"[{category}] SEC Form {form_type} - {company_name} ({ticker})"
                
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
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key or api_key == "***":
        print("ERROR: RESEND_API_KEY is missing or invalid. Check your .env file.")
        print("  - Ensure RESEND_API_KEY is set (e.g. RESEND_API_KEY=re_xxxxx)")
        print("  - Do not mask or replace the value with '***'.")
        logging.error("RESEND_API_KEY not found or masked; cannot send email.")
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

    # Load SEC company tickers mapping (cached for session)
    try:
        logging.info("Loading SEC company tickers mapping...")
        ticker_cik_mapping = get_company_tickers_mapping()
        logging.info("Loaded %d ticker-to-CIK mappings.", len(ticker_cik_mapping))
    except Exception as exc:
        logging.critical("Failed to load SEC ticker mapping: %s", exc, exc_info=True)
        return

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

    # Process tickers sequentially (one at a time) for Gemini rate limit compliance
    print(f"\n{'='*60}")
    print(f"Processing {len(ticker_to_name)} tickers sequentially...")
    print(f"{'='*60}")

    for ticker, company_name in ticker_to_name.items():
        try:
            summaries = process_ticker(ticker, company_name, ticker_cik_mapping, supabase, gemini_client)
            all_summaries.extend(summaries)
            print(f"✓ Completed: {company_name} ({ticker}) - {len(summaries)} summary(ies)")
        except Exception as exc:
            logging.error(
                "Unexpected error while processing ticker %s: %s",
                ticker,
                exc,
                exc_info=True,
            )
            all_summaries.append({
                "ticker": ticker,
                "company_name": company_name,
                "summary": "No news.",
                "source_url": None,
                "is_no_news": True,
            })

    # TICKER PROCESSING FINISHED - Now sending email
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL PROCESSING COMPLETED. Processed {len(ticker_to_name)} tickers.")
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
