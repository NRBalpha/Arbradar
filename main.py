from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import string
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

try:
    import anthropic as _anthropic_mod
except ImportError:
    _anthropic_mod = None

load_dotenv()

# ── config ──────────────────────────────────────────────────────────────────

MIN_GAP = float(os.getenv("MIN_GAP", "1.5"))
MIN_VOLUME = float(os.getenv("MIN_VOLUME", "500"))
TIMEOUT = 10
RETRY_DELAY = 2
START_TIME = time.time()
CACHE_TTL = 30  # seconds
BASE = Path(__file__).parent
HISTORY_FILE = BASE / "arb_history.json"
ERROR_LOG = BASE / "errors.log"
WATCHLIST_FILE = BASE / "watchlist.json"
SETTINGS_FILE = BASE / "settings.json"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
METACULUS_API_TOKEN = os.getenv("METACULUS_API_TOKEN", "")

_ai_client = None
if _anthropic_mod and ANTHROPIC_API_KEY:
    try:
        _ai_client = _anthropic_mod.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:
        pass

_ai_cache: dict = {}
_news_cache: dict = {}
_req_metrics = {"total": 0, "cache_hits": 0, "cache_misses": 0, "errors": []}

PLATFORM_FEES = {
    "poly": 0.02,
    "predict": 0.10,
    "manifold": 0.0,
    "metaculus": 0.0,
    "opinion": 0.02,
}

STOP_WORDS = frozenset(
    {"the", "a", "an", "in", "at", "of", "on", "for", "to", "by", "will", "be", "is",
     "and", "or", "it", "this", "that", "with", "from", "are", "was", "were", "has",
     "have", "had", "do", "does", "did", "but", "not", "so", "if", "than", "its"}
)

EXPANDED_STOPWORDS = frozenset({
    "the", "a", "an", "in", "at", "of", "on", "for", "to", "by", "will",
    "be", "is", "are", "was", "were", "has", "have", "had", "that", "this",
    "and", "or", "it", "with", "from", "do", "does", "did", "but", "not",
    "so", "if", "than", "its", "can", "could", "would", "should", "may",
    "might", "shall", "must", "between", "within", "during",
    "before", "after", "about", "into", "through", "market", "price",
    "who", "what", "when", "where", "how", "which", "their", "there",
    "more", "less", "first", "last", "next", "new", "old", "end",
    "yes", "no", "lose", "loses", "result", "results",
})

KEYWORD_SYNONYMS = {
    "republican": "republican",
    "gop": "republican",
    "democrat": "democrat",
    "democratic": "democrat",
    "senate": "senate",
    "senator": "senate",
    "win": "win",
    "wins": "win",
    "winner": "win",
    "election": "election",
    "vote": "election",
    "voting": "election",
}

US_STATES = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california",
    "colorado", "connecticut", "delaware", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
    "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
})


# ── logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arbdar")

file_handler = logging.FileHandler(ERROR_LOG)
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(file_handler)

# ── app ─────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup diagnostics: test each platform, print status table."""
    import sys
    p = lambda s: (print(s), sys.stdout.flush())
    p("")
    p("=" * 50)
    p("  ArbRadar v3 — Starting up")
    p("=" * 50)

    platforms = [
        ("Polymarket", "poly", fetch_polymarket),
        ("PredictIt", "predict", fetch_predictit),
        ("Manifold", "manifold", fetch_manifold),
        ("Metaculus", "metaculus", fetch_metaculus),
        ("Opinion", "opinion", fetch_opinion),
    ]

    total = 0
    for name, key, func in platforms:
        try:
            result = await asyncio.wait_for(func(), timeout=12)
            count = len(result) if result else 0
            total += count
            status = "\u2713" if count > 0 else "\u2717"
            p(f"  {status} {name:<15} {count} markets")
            _platform_status[key] = {
                "status": "live" if count > 0 else "down",
                "count": count,
                "latency_ms": None,
                "last_success": datetime.now(timezone.utc).isoformat() if count > 0 else None,
                "error": None if count > 0 else "no data",
            }
        except asyncio.TimeoutError:
            p(f"  \u2717 {name:<15} TIMEOUT")
            _platform_status[key] = {
                "status": "down", "count": 0, "latency_ms": None,
                "last_success": None, "error": "timeout",
            }
        except Exception as e:
            p(f"  \u2717 {name:<15} ERROR: {str(e)[:40]}")
            _platform_status[key] = {
                "status": "down", "count": 0, "latency_ms": None,
                "last_success": None, "error": str(e)[:200],
            }

    p(f"\n  Total: {total} markets loaded")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key and api_key not in ("", "your_key_here"):
        p("  \u2713 Anthropic API key configured")
    else:
        p("  \u2717 No Anthropic API key — AI features disabled")
        p("    Add ANTHROPIC_API_KEY to your .env file")

    if not METACULUS_API_TOKEN:
        p("  \u2139 Metaculus requires API token (get from metaculus.com/aib)")
        p("    Add METACULUS_API_TOKEN to .env for Metaculus data")

    p("=" * 50)
    p("  http://localhost:8000")
    p("=" * 50 + "\n")

    yield

    print("\nArbRadar shutting down...")


app = FastAPI(title="ArbRadar", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ─────────────────────────────────────────────────────────────────


def clean_title(title: str) -> set:
    title = title.lower().translate(str.maketrans("", "", string.punctuation))
    return set(title.split()) - STOP_WORDS


def extract_keywords(title: str) -> set:
    """Improved keyword extraction with expanded stopwords and synonym normalization."""
    t = re.sub(r'[^a-z0-9\s]', ' ', title.lower())
    return {KEYWORD_SYNONYMS.get(w, w) for w in t.split() if w not in EXPANDED_STOPWORDS and len(w) > 2}


def extract_numbers(title: str) -> set:
    """Extract years, percentages, dollar amounts from title."""
    return set(re.findall(r'\b\d+(?:\.\d+)?%?(?:k|m|b)?\b', title.lower()))


_NON_ENTITY_WORDS = frozenset({
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "Will", "What", "How", "Who", "When", "Where", "Which", "Why",
    "Before", "After", "Between", "During", "Through", "Under", "Over",
    "The", "And", "But", "Not", "Are", "Can", "Could", "Would",
    "Should", "Does", "Did", "Has", "Had", "Have", "Many", "More",
    "Than", "This", "That", "Each", "Some", "Any", "End", "New",
    # Common capitalized nouns that aren't discriminating entities
    "World", "Cup", "League", "Series", "Bowl", "Finals", "Championship",
    "Election", "Senate", "House", "President", "Party", "Rate", "Price",
    "Federal", "Reserve", "National", "United", "States", "Average",
})

def extract_proper_nouns(title: str) -> set:
    """Extract capitalized words and acronyms that are likely entity names."""
    # Match: Capitalized words (Iran, Trump) and all-caps acronyms (USA, FIFA, NBA)
    words = set()
    for w in re.findall(r'\b[A-Z][a-z]{2,}\b', title):
        if w not in _NON_ENTITY_WORDS:
            words.add(w)
    _ACRONYM_EXCLUDE = {
        "OR", "AND", "THE", "NOT", "YES", "NO", "IF", "BY", "AT", "IN", "ON", "TO", "US",
        # League/org names are categories, not discriminating entities
        "FIFA", "NBA", "NFL", "MLB", "NHL", "UFC", "MLS", "PGA", "ATP", "WTA",
        "EU", "UN", "WHO", "IMF", "SEC", "FED", "GDP", "IPO", "CEO", "AI",
        "FOMC", "CPI", "GDP", "ETF", "BTC", "ETH",
    }
    for w in re.findall(r'\b[A-Z]{2,6}\b', title):
        if w not in _ACRONYM_EXCLUDE:
            words.add(w)
    return words


def _extract_us_states(title: str) -> set:
    """Extract US state names from a title using word-boundary matching."""
    t = title.lower()
    found = set()
    for state in US_STATES:
        if re.search(r'\b' + re.escape(state) + r'\b', t):
            found.add(state)
    # "west virginia" subsumes "virginia"
    if "west virginia" in found:
        found.discard("virginia")
    return found


def _strip_contract_suffix(title: str) -> str:
    """Strip PredictIt contract suffixes like ' — Democratic', ' — Republican'."""
    idx = title.find(" \u2014 ")
    if idx != -1:
        return title[:idx].strip()
    return title


def _normalize_base_title(title: str) -> str:
    """Normalize to base title for dedup: strip after ' — ', remove party/state suffixes."""
    # Strip everything after em dash (PredictIt contract suffix)
    idx = title.find(" \u2014 ")
    if idx != -1:
        title = title[:idx]

    t = title.strip()
    t_lower = t.lower()

    # Strip trailing party labels
    for suffix in ["republican", "democrat", "democratic", "gop",
                    "libertarian", "independent", "green party"]:
        if t_lower.endswith(suffix):
            t = t[:len(t) - len(suffix)].rstrip(" -\u2013\u2014,:")
            t_lower = t.lower()

    # Strip trailing state names (longest first to match "west virginia" before "virginia")
    for state in sorted(US_STATES, key=len, reverse=True):
        if t_lower.endswith(state):
            t = t[:len(t) - len(state)].rstrip(" -\u2013\u2014,:")
            t_lower = t.lower()

    # Final normalization: lowercase, strip punctuation, sort keywords
    t = re.sub(r'[^a-z0-9\s]', ' ', t_lower)
    words = sorted(w for w in t.split() if w not in EXPANDED_STOPWORDS and len(w) > 2)
    return ' '.join(words)


def title_similarity(a: str, b: str) -> float:
    """Original Jaccard on basic cleaned titles (used by scorer)."""
    sa, sb = clean_title(a), clean_title(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def keyword_similarity(a: str, b: str) -> float:
    """Improved Jaccard on keyword-extracted titles."""
    sa, sb = extract_keywords(a), extract_keywords(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def entity_match(a: str, b: str) -> bool:
    """Fallback: match if 2+ numbers AND 1+ proper noun overlap."""
    nums_a, nums_b = extract_numbers(a), extract_numbers(b)
    nouns_a, nouns_b = extract_proper_nouns(a), extract_proper_nouns(b)
    num_overlap = nums_a & nums_b
    noun_overlap = nouns_a & nouns_b
    return len(num_overlap) >= 2 and len(noun_overlap) >= 1


def normalize_title(title: str) -> str:
    """Normalize title for dedup: lowercase, remove punctuation, sort keywords."""
    t = re.sub(r'[^a-z0-9\s]', ' ', title.lower())
    words = sorted(w for w in t.split() if w not in EXPANDED_STOPWORDS and len(w) > 2)
    return ' '.join(words)


BAD_TAIL_PATTERNS = re.compile(
    r'^(democratic|republican|yes|no|other|none|draw|tie|over|under|'
    r'6 or 7|7 or more|5 or fewer|male|female|man|woman)$',
    re.IGNORECASE,
)


def is_cross_match(a: str, b: str) -> tuple[bool, float]:
    """Check if two titles match across platforms. Returns (matched, sim)."""
    # Strip PredictIt contract suffixes (e.g. " — Democratic") before comparing
    a_clean = _strip_contract_suffix(a)
    b_clean = _strip_contract_suffix(b)

    sim = keyword_similarity(a_clean, b_clean)
    if sim < 0.40:
        return False, sim

    # Entity consistency: if both titles have unique proper nouns
    # that don't overlap, this is likely a different event (e.g. Iran vs Canada)
    nouns_a = extract_proper_nouns(a_clean)
    nouns_b = extract_proper_nouns(b_clean)
    if nouns_a and nouns_b:
        unique_a = nouns_a - nouns_b
        unique_b = nouns_b - nouns_a
        shared = nouns_a & nouns_b
        # If both have unique entities and no shared entities, reject
        if unique_a and unique_b and not shared:
            return False, sim

    if sim >= 0.40:
        return True, sim
    return False, sim


def parse_dt(s: Any) -> datetime | None:
    if not s:
        return None
    if isinstance(s, (int, float)):
        try:
            return datetime.fromtimestamp(s / 1000 if s > 1e12 else s, tz=timezone.utc)
        except Exception:
            return None
    try:
        ts = str(s).replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def days_until(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    delta = dt - datetime.now(timezone.utc)
    return max(int(delta.total_seconds() / 86400), 0)


def load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def save_history(data: list):
    HISTORY_FILE.write_text(json.dumps(data, indent=2, default=str))


def load_watchlist() -> list:
    if WATCHLIST_FILE.exists():
        try:
            return json.loads(WATCHLIST_FILE.read_text())
        except Exception:
            return []
    return []


def save_watchlist(data: list):
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2, default=str))


def load_settings() -> dict:
    defaults = {
        "bankroll": 500, "min_gap": 1.5, "min_volume": 0,
        "half_kelly": True, "alert_threshold": 5,
        "notifications": True, "auto_refresh": 90,
        "platforms": {"poly": True, "predict": True, "manifold": True, "metaculus": True, "opinion": True},
    }
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text())
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def save_settings(data: dict):
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


# ── platform fetchers ───────────────────────────────────────────────────────


async def fetch_polymarket() -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://gamma-api.polymarket.com/markets"
            "?limit=100&active=true&closed=false"
        )
        resp.raise_for_status()
        data = resp.json()

    if not isinstance(data, list):
        data = []

    markets = []
    for m in data:
        try:
            question = m.get("question", "")
            if not question:
                continue

            raw_prices = m.get("outcomePrices", "[]")
            prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            if not prices:
                continue
            yes_price = float(prices[0])
            no_price = float(prices[1]) if len(prices) > 1 else round(1 - yes_price, 4)

            if yes_price <= 0.005 or yes_price >= 0.995:
                continue

            slug = m.get("slug", m.get("id", ""))
            url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com/markets"
            end_date = m.get("endDate") or m.get("end_date_iso")
            close_time = parse_dt(end_date)

            raw_outcomes = m.get("outcomes", "[]")
            outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes

            markets.append({
                "id": str(m.get("conditionId", m.get("id", ""))),
                "title": question,
                "probability": round(yes_price * 100, 2),
                "volume": float(m.get("volume", 0) or 0),
                "platform": "poly",
                "url": url,
                "yes_price": round(yes_price, 4),
                "no_price": round(no_price, 4),
                "close_time": close_time.isoformat() if close_time else None,
                "slug": slug,
                "outcomes": outcomes if isinstance(outcomes, list) else [],
                "event_id": m.get("questionID", m.get("question_id", slug)),
            })
        except Exception:
            continue
    logger.info(f"[polymarket] {len(markets)} markets")
    return markets


async def fetch_predictit() -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get("https://www.predictit.org/api/marketdata/all/")
        resp.raise_for_status()
        data = resp.json()

    markets = []
    for market in data.get("markets", []):
        market_name = market.get("name", "Unknown")
        market_id = market.get("id", "")

        for contract in market.get("contracts", []):
            contract_name = contract.get("name", "")

            # Build meaningful title
            if contract_name and contract_name != market_name:
                full_title = f"{market_name} — {contract_name}"
            else:
                full_title = market_name

            # Skip garbage titles
            if len(full_title.split()) < 3:
                continue

            yes_bid = contract.get("bestBuyYesCost") or contract.get("bestYesBid") or 0
            no_bid = contract.get("bestBuyNoCost") or contract.get("bestNoBid") or 0
            last_trade = contract.get("lastTradePrice") or 0

            # Need at least some price signal
            if yes_bid == 0 and no_bid == 0 and last_trade == 0:
                continue

            best_yes = yes_bid or last_trade
            prob = round(float(best_yes) * 100, 1) if best_yes else round((1 - float(no_bid)) * 100, 1)

            end_date = contract.get("dateEnd") or market.get("dateEnd")
            close_time = parse_dt(end_date)

            markets.append({
                "id": str(contract.get("id", "")),
                "title": full_title,
                "probability": prob,
                "volume": contract.get("volume") or market.get("totalSharesTraded", 0) or 0,
                "platform": "predict",
                "url": f"https://www.predictit.org/markets/detail/{market_id}",
                "yes_price": float(best_yes) if best_yes else None,
                "no_price": float(no_bid) if no_bid else None,
                "close_time": close_time.isoformat() if close_time else None,
                "market_id": str(market_id),
            })
    logger.info(f"[predictit] {len(markets)} contracts")
    return markets


async def fetch_manifold() -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://api.manifold.markets/v0/markets"
            "?limit=100&sort=last-bet-time"
        )
        resp.raise_for_status()
        data = resp.json()

    markets = []
    for m in data if isinstance(data, list) else []:
        if m.get("outcomeType") != "BINARY" or m.get("isResolved", False):
            continue
        prob = m.get("probability", 0.5)
        close_time = parse_dt(m.get("closeTime"))
        markets.append({
            "id": str(m.get("id", "")),
            "title": m.get("question", "Unknown"),
            "probability": round(float(prob) * 100, 2),
            "volume": float(m.get("volume", 0) or 0),
            "platform": "manifold",
            "url": m.get("url", "https://manifold.markets"),
            "close_time": close_time.isoformat() if close_time else None,
        })
    logger.info(f"[manifold] {len(markets)} markets")
    return markets


async def fetch_metaculus() -> list[dict]:
    # Metaculus API requires authentication since 2025
    metaculus_token = METACULUS_API_TOKEN

    urls_to_try = [
        "https://www.metaculus.com/api2/questions/?limit=50&status=open&type=forecast&order_by=-activity",
        "https://www.metaculus.com/api2/questions/?limit=30&status=open",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }
    if metaculus_token:
        headers["Authorization"] = f"Token {metaculus_token}"

    for api_url in urls_to_try:
        try:
            async with httpx.AsyncClient(
                timeout=15,
                headers=headers,
                follow_redirects=True,
            ) as client:
                resp = await client.get(api_url)
                if resp.status_code != 200:
                    logger.info(f"[metaculus] {api_url} returned {resp.status_code}")
                    continue

                data = resp.json()

                # Handle both list and dict with "results" key
                if isinstance(data, list):
                    results = data
                elif isinstance(data, dict):
                    results = data.get("results", [])
                else:
                    continue

                markets = []
                for q in results:
                    prob = None

                    # Path 1: community_prediction.full.q2 / mean
                    cp = q.get("community_prediction", {})
                    if isinstance(cp, dict):
                        full = cp.get("full", {})
                        if isinstance(full, dict):
                            prob = full.get("q2") or full.get("mean")
                    # Path 2: direct probability / forecasts field
                    if prob is None:
                        prob = q.get("community_prediction_at_access_time")
                    if prob is None:
                        prob = q.get("prediction_count") and q.get("community_prediction")
                        if isinstance(prob, dict):
                            prob = None

                    if prob is None:
                        continue

                    try:
                        prob_f = float(prob)
                        prob_pct = round(prob_f * 100, 1) if prob_f <= 1 else round(prob_f, 1)
                    except (ValueError, TypeError):
                        continue

                    if not (0 <= prob_pct <= 100):
                        continue

                    question_id = q.get("id", "")
                    page_url = q.get("page_url") or q.get("url") or ""
                    if page_url and not page_url.startswith("http"):
                        url_full = f"https://www.metaculus.com{page_url}"
                    elif page_url:
                        url_full = page_url
                    else:
                        url_full = f"https://www.metaculus.com/questions/{question_id}/"

                    close_time = parse_dt(q.get("resolve_time") or q.get("close_time") or q.get("scheduled_close_time"))

                    markets.append({
                        "id": f"meta_{question_id}",
                        "title": q.get("title", q.get("question", "Unknown")),
                        "probability": prob_pct,
                        "volume": q.get("number_of_forecasters", 0) or q.get("forecasts_count", 0) or 0,
                        "platform": "metaculus",
                        "url": url_full,
                        "close_time": close_time.isoformat() if close_time else None,
                    })

                if markets:
                    logger.info(f"[metaculus] {len(markets)} markets from {api_url}")
                    return markets

        except Exception as e:
            logger.warning(f"[metaculus] {api_url} failed: {e}")
            continue

    logger.warning("[metaculus] all attempts failed, returning []")
    return []


async def fetch_opinion() -> list[dict]:
    endpoints = [
        "https://opinionmarkets.io/api/markets",
        "https://api.opinion.markets/v1/markets",
        "https://opinion.markets/api/markets",
        "https://opinionmarkets.com/api/markets",
        "https://opinionmarkets.io/api/v1/markets",
        "https://opinionmarkets.io/api/markets?status=open",
    ]

    for ep_url in endpoints:
        try:
            async with httpx.AsyncClient(
                timeout=12,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json",
                },
            ) as client:
                resp = await client.get(ep_url)
                if resp.status_code not in (200, 201):
                    continue

                raw = resp.json()

                # Handle different response shapes
                if isinstance(raw, list):
                    items = raw
                elif isinstance(raw, dict):
                    items = (
                        raw.get("markets")
                        or raw.get("data")
                        or raw.get("results")
                        or raw.get("items")
                        or []
                    )
                else:
                    continue

                markets = []
                for m in items if isinstance(items, list) else []:
                    if not isinstance(m, dict):
                        continue

                    prob = (
                        m.get("probability")
                        or m.get("yes_price")
                        or m.get("price")
                        or m.get("lastTradePrice")
                        or 0.5
                    )
                    try:
                        prob = float(prob)
                        if prob > 1:
                            prob = prob / 100
                        prob_pct = round(prob * 100, 1)
                    except (ValueError, TypeError):
                        prob_pct = 50.0

                    title = m.get("title") or m.get("question") or m.get("name") or "Unknown"
                    url_field = m.get("url") or m.get("link") or m.get("market_url") or "https://opinionmarkets.io"

                    markets.append({
                        "id": f"opn_{m.get('id', len(markets))}",
                        "title": title,
                        "probability": prob_pct,
                        "volume": float(m.get("volume") or m.get("total_volume") or 0),
                        "platform": "opinion",
                        "url": url_field,
                        "close_time": m.get("close_time") or m.get("end_date") or m.get("resolution_date"),
                    })

                if markets:
                    logger.info(f"[opinion] {len(markets)} markets from {ep_url}")
                    return markets

        except Exception as e:
            logger.warning(f"[opinion] {ep_url} failed: {e}")
            continue

    logger.warning("[opinion] no working endpoint found")
    return []


# ── safe wrappers ───────────────────────────────────────────────────────────

FETCHERS = {
    "poly": ("polymarket", fetch_polymarket),
    "predict": ("predictit", fetch_predictit),
    "manifold": ("manifold", fetch_manifold),
    "metaculus": ("metaculus", fetch_metaculus),
    "opinion": ("opinion", fetch_opinion),
}

# Per-platform status tracking for /health
_platform_status: dict[str, dict] = {
    k: {"status": "down", "count": 0, "latency_ms": None, "last_success": None, "error": None}
    for k in FETCHERS
}


async def safe_fetch(name: str, fn) -> tuple[list[dict], str | None]:
    """Fetch with retry: try twice before giving up."""
    t0 = time.time()
    for attempt in range(2):
        try:
            result = await fn()
            latency = round((time.time() - t0) * 1000)
            if result:
                _platform_status[name] = {
                    "status": "live",
                    "count": len(result),
                    "latency_ms": latency,
                    "last_success": datetime.now(timezone.utc).isoformat(),
                    "error": None,
                }
                return result, None
            # Empty result — retry once
            if attempt == 0:
                logger.info(f"[{name}] empty result, retrying...")
                await asyncio.sleep(2)
        except Exception as e:
            latency = round((time.time() - t0) * 1000)
            if attempt == 0:
                logger.info(f"[{name}] attempt 1 failed ({e}), retrying...")
                await asyncio.sleep(2)
            else:
                msg = f"[{name}] {type(e).__name__}: {e}"
                logger.error(msg)
                _platform_status[name] = {
                    "status": "down",
                    "count": 0,
                    "latency_ms": latency,
                    "last_success": _platform_status[name].get("last_success"),
                    "error": str(e)[:200],
                }
                return [], msg

    # Both attempts returned empty
    latency = round((time.time() - t0) * 1000)
    _platform_status[name] = {
        "status": "down",
        "count": 0,
        "latency_ms": latency,
        "last_success": _platform_status[name].get("last_success"),
        "error": "empty result after 2 attempts",
    }
    return [], f"[{name}] empty result after retries"


# ── market endpoints ────────────────────────────────────────────────────────

@app.get("/market/poly/search")
async def search_polymarket(q: str = Query(...)):
    """Search Polymarket Gamma API for a specific market by name."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                f"https://gamma-api.polymarket.com/markets?search={q}&limit=5&active=true"
            )
            resp.raise_for_status()
            results = resp.json()
            return [
                {
                    "title": r.get("question", ""),
                    "url": f"https://polymarket.com/event/{r.get('slug', '')}",
                    "slug": r.get("slug", ""),
                    "probability": round(float(json.loads(r.get("outcomePrices", "[0.5]"))[0]) * 100, 1)
                        if r.get("outcomePrices") else 50,
                    "volume": float(r.get("volume", 0) or 0),
                }
                for r in (results if isinstance(results, list) else [])
                if r.get("slug")
            ]
    except Exception as e:
        logger.error(f"[poly search] {e}")
        return []


@app.get("/markets/polymarket")
async def ep_polymarket():
    data, _ = await safe_fetch("poly", fetch_polymarket)
    return data


@app.get("/markets/predictit")
async def ep_predictit():
    data, _ = await safe_fetch("predict", fetch_predictit)
    return data


@app.get("/markets/manifold")
async def ep_manifold():
    data, _ = await safe_fetch("manifold", fetch_manifold)
    return data


@app.get("/markets/metaculus")
async def ep_metaculus():
    data, _ = await safe_fetch("metaculus", fetch_metaculus)
    return data


@app.get("/markets/opinion")
async def ep_opinion():
    data, _ = await safe_fetch("opinion", fetch_opinion)
    return data


_market_cache: dict = {"data": None, "ts": 0}


async def _fetch_all_markets() -> dict:
    now = time.time()
    if _market_cache["data"] and now - _market_cache["ts"] < CACHE_TTL:
        return _market_cache["data"]

    platform_names = list(FETCHERS.keys())
    coros = [safe_fetch(k, v[1]) for k, v in FETCHERS.items()]
    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    all_markets = []
    counts = {}
    errors = []
    for i, res in enumerate(raw_results):
        name = platform_names[i]
        if isinstance(res, Exception):
            errors.append({"platform": name, "error_msg": str(res)})
            counts[name] = 0
            _platform_status[name] = {
                "status": "down", "count": 0, "latency_ms": None,
                "last_success": _platform_status[name].get("last_success"),
                "error": str(res)[:200],
            }
        elif isinstance(res, tuple):
            data, err = res
            all_markets.extend(data)
            counts[name] = len(data)
            if err:
                errors.append({"platform": name, "error_msg": err})
                # partial if we got some data despite an error
                if len(data) > 0 and _platform_status[name].get("status") != "live":
                    _platform_status[name]["status"] = "partial"
        else:
            counts[name] = 0

    result = {
        "markets": all_markets,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "errors": errors,
    }
    _market_cache["data"] = result
    _market_cache["ts"] = now
    return result


@app.get("/markets/all")
async def markets_all():
    return await _fetch_all_markets()


# ── arb detection: cross-platform ──────────────────────────────────────────

def compute_fee_adjusted_gap(prob_a: float, prob_b: float, plat_a: str, plat_b: str) -> float:
    raw_gap = abs(prob_a - prob_b)
    fee_a = PLATFORM_FEES.get(plat_a, 0.02) * 100
    fee_b = PLATFORM_FEES.get(plat_b, 0.02) * 100
    return raw_gap - fee_a - fee_b


def compute_annualized(gap_pct: float, days: int | None) -> float | None:
    if days is None:
        return None
    d = max(days, 1)
    return round((gap_pct / 100) / d * 365 * 100, 2)


@app.get("/arb/cross")
async def arb_cross(min_gap: float = Query(default=1.5), min_volume: float = Query(default=0)):
    data = await _fetch_all_markets()
    markets = data["markets"]

    by_platform: dict[str, list[dict]] = {}
    for m in markets:
        by_platform.setdefault(m["platform"], []).append(m)

    platforms = list(by_platform.keys())
    arbs = []
    seen_pairs: set[tuple] = set()

    # Filtering counters
    total_considered = 0
    removed_gap = 0
    removed_state = 0
    removed_num = 0

    for i in range(len(platforms)):
        for j in range(i + 1, len(platforms)):
            for a in by_platform[platforms[i]]:
                for b in by_platform[platforms[j]]:
                    matched, sim = is_cross_match(a["title"], b["title"])
                    if not matched:
                        continue

                    total_considered += 1

                    # Strip suffixes for state/number checks
                    a_clean = _strip_contract_suffix(a["title"])
                    b_clean = _strip_contract_suffix(b["title"])

                    # Fix 1c: State mismatch — different states = different races
                    states_a = _extract_us_states(a_clean)
                    states_b = _extract_us_states(b_clean)
                    if states_a and states_b and not (states_a & states_b):
                        removed_state += 1
                        continue

                    # Fix 2: 4-digit number mismatch — different years = different events
                    years_a = set(re.findall(r'\b\d{4}\b', a_clean))
                    years_b = set(re.findall(r'\b\d{4}\b', b_clean))
                    if years_a and years_b and not (years_a & years_b):
                        removed_num += 1
                        continue

                    raw_gap = abs(a["probability"] - b["probability"])

                    # Fix 1a: Gap > 55% means completely different markets
                    if raw_gap > 55:
                        removed_gap += 1
                        continue

                    fee_gap = compute_fee_adjusted_gap(
                        a["probability"], b["probability"],
                        a["platform"], b["platform"],
                    )
                    if raw_gap < min_gap:
                        continue
                    min_vol = min(a.get("volume", 0) or 0, b.get("volume", 0) or 0)
                    if min_vol < min_volume:
                        continue

                    if a["probability"] >= b["probability"]:
                        high, low = a, b
                    else:
                        high, low = b, a

                    # Fix 1b: Extreme probability mismatch = different markets
                    if high["probability"] > 88 and low["probability"] < 12:
                        removed_gap += 1
                        continue

                    # Deduplicate: skip if same pair of (id+platform) already seen
                    pair_key = tuple(sorted([
                        high["id"] + high["platform"],
                        low["id"] + low["platform"],
                    ]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    ct_high = parse_dt(high.get("close_time"))
                    ct_low = parse_dt(low.get("close_time"))
                    ct = ct_high or ct_low
                    dtr = days_until(ct)
                    ann = compute_annualized(fee_gap, dtr)

                    # Fix 4: Quality confidence flag
                    if raw_gap < 15 and sim > 0.6:
                        confidence = "high"
                    elif raw_gap < 30 and sim > 0.45:
                        confidence = "medium"
                    else:
                        confidence = "low"

                    arbs.append({
                        "title": high["title"],
                        "title_low": low["title"],
                        "platform_high": high["platform"],
                        "platform_low": low["platform"],
                        "prob_high": high["probability"],
                        "prob_low": low["probability"],
                        "raw_gap": round(raw_gap, 2),
                        "fee_adjusted_gap": round(fee_gap, 2),
                        "min_volume": min_vol,
                        "days_to_resolution": dtr,
                        "annualized_return": ann,
                        "url_high": high.get("url", ""),
                        "url_low": low.get("url", ""),
                        "slug_high": high.get("slug", ""),
                        "slug_low": low.get("slug", ""),
                        "similarity": round(sim, 3),
                        "volume_high": high.get("volume", 0),
                        "volume_low": low.get("volume", 0),
                        "confidence": confidence,
                    })

    before_dedup = len(arbs)

    # Filter out negative fee-adjusted gap (not real arb after fees)
    arbs = [a for a in arbs if a["fee_adjusted_gap"] > 0]

    # Deduplicate: when the same two market URLs appear multiple times
    # (e.g., PredictIt contracts from the same parent market all matching
    # the same Manifold market), keep only the highest-similarity match.
    url_dedup: dict[tuple, dict] = {}
    for arb in arbs:
        key = tuple(sorted([arb["url_high"], arb["url_low"]]))
        existing = url_dedup.get(key)
        if existing is None or arb["similarity"] > existing["similarity"]:
            url_dedup[key] = arb
    arbs = list(url_dedup.values())

    # Normalize-title dedup: combine both titles for the key so
    # the same event pair is caught regardless of which side is "high"
    norm_dedup: dict[str, dict] = {}
    for arb in arbs:
        parts = sorted([normalize_title(arb["title"]), normalize_title(arb.get("title_low", ""))])
        nkey = "|".join(parts)
        existing = norm_dedup.get(nkey)
        if existing is None or arb["fee_adjusted_gap"] > existing["fee_adjusted_gap"]:
            norm_dedup[nkey] = arb
    arbs = list(norm_dedup.values())

    # Remove near-duplicate arbs (>0.7 keyword overlap on EITHER title),
    # keep the one with the higher fee-adjusted gap
    final: list[dict] = []
    for arb in sorted(arbs, key=lambda x: x["fee_adjusted_gap"], reverse=True):
        is_dup = False
        for f in final:
            if (keyword_similarity(arb["title"], f["title"]) > 0.7
                    or keyword_similarity(arb.get("title_low", ""), f.get("title_low", "")) > 0.7
                    or keyword_similarity(arb["title"], f.get("title_low", "")) > 0.7
                    or keyword_similarity(arb.get("title_low", ""), f["title"]) > 0.7):
                is_dup = True
                break
        if not is_dup:
            final.append(arb)
    arbs = final

    # ── Dedup step 1: Group by normalized base title, keep best per group ──
    # Strips " — <contract>" suffixes and state/party suffixes so that
    # e.g. "Senate race — Republican" and "Senate race — Democratic" collapse.
    base_dedup: dict[str, dict] = {}
    for arb in arbs:
        base_key = _normalize_base_title(arb["title"])
        existing = base_dedup.get(base_key)
        if existing is None or arb["fee_adjusted_gap"] > existing["fee_adjusted_gap"]:
            base_dedup[base_key] = arb
    arbs = list(base_dedup.values())

    # ── Dedup step 2: Deduplicate by market pair (platform combo + similar title) ──
    # If platform_a + platform_b + similar base title appears more than once,
    # keep only the one with the highest fee_adjusted_gap.
    pair_dedup: dict[tuple, dict] = {}
    for arb in arbs:
        plat_pair = tuple(sorted([arb["platform_high"], arb["platform_low"]]))
        base = _normalize_base_title(arb["title"])
        key = (plat_pair[0], plat_pair[1], base)
        existing = pair_dedup.get(key)
        if existing is None or arb["fee_adjusted_gap"] > existing["fee_adjusted_gap"]:
            pair_dedup[key] = arb
    arbs = list(pair_dedup.values())

    # ── Dedup step 3: No duplicate market titles in the final list ──
    # Once a market title (either side) is used, it cannot appear again.
    seen_base_titles: set[str] = set()
    unique_arbs: list[dict] = []
    for arb in sorted(arbs, key=lambda x: x["fee_adjusted_gap"], reverse=True):
        base_high = _normalize_base_title(arb["title"])
        base_low = _normalize_base_title(arb.get("title_low", ""))
        if base_high in seen_base_titles or (base_low and base_low in seen_base_titles):
            continue
        seen_base_titles.add(base_high)
        if base_low:
            seen_base_titles.add(base_low)
        unique_arbs.append(arb)
    arbs = unique_arbs

    arbs.sort(key=lambda x: x["fee_adjusted_gap"], reverse=True)

    removed_dup = before_dedup - len(arbs)

    print(f"Total pairs before filtering: {total_considered}")
    print(f"Removed - gap too large: {removed_gap}")
    print(f"Removed - state mismatch: {removed_state}")
    print(f"Removed - number mismatch: {removed_num}")
    print(f"Removed - duplicates: {removed_dup}")
    print(f"Final real arbs: {len(arbs)}")

    return arbs


# ── arb detection: dutch-book ──────────────────────────────────────────────

@app.get("/arb/dutch")
async def arb_dutch(min_edge: float = Query(default=1)):
    poly_result, pi_result = await asyncio.gather(
        safe_fetch("polymarket", fetch_polymarket),
        safe_fetch("predictit", fetch_predictit),
    )
    poly_markets = poly_result[0]
    pi_markets = pi_result[0]

    # Group by event_id first (Polymarket)
    events: dict[str, list[dict]] = {}
    ungrouped = []
    for m in poly_markets:
        eid = m.get("event_id")
        if eid and eid != m.get("slug") and eid != m.get("id"):
            events.setdefault(eid, []).append(m)
        else:
            ungrouped.append(m)

    # Group PredictIt contracts by market_id (contracts in same market are mutually exclusive)
    pi_by_market: dict[str, list[dict]] = {}
    for m in pi_markets:
        mid = m.get("market_id", "")
        if mid:
            pi_by_market.setdefault("pi-" + mid, []).append(m)
    events.update(pi_by_market)

    # Fuzzy group ungrouped Polymarket markets by first 60 chars prefix
    prefix_groups: dict[str, list[dict]] = {}
    for m in ungrouped:
        prefix = m["title"][:60].lower().strip()
        placed = False
        # Try event_id groups first
        for eid, group in events.items():
            if keyword_similarity(m["title"], group[0]["title"]) > 0.6:
                group.append(m)
                placed = True
                break
        if not placed:
            # Group by prefix
            for pkey, pgroup in prefix_groups.items():
                if keyword_similarity(m["title"], pgroup[0]["title"]) > 0.6:
                    pgroup.append(m)
                    placed = True
                    break
            if not placed:
                prefix_groups[prefix] = [m]

    # Merge prefix groups into events
    for pkey, pgroup in prefix_groups.items():
        events["prefix-" + pkey[:30]] = pgroup

    results = []
    for eid, group in events.items():
        if len(group) < 2:
            continue
        sum_probs = sum((m.get("yes_price") or 0) for m in group)
        if sum_probs <= 0:
            continue
        # No profit if total prices >= 1.0
        if sum_probs >= 1.0:
            continue
        if sum_probs < 0.985:
            edge = (1 - sum_probs) * 100
            if edge < min_edge:
                continue

            guaranteed_profit_per_dollar = 1 - sum_probs
            # Profit cannot exceed amount invested; if it does, data is bad
            invalid_arb = False
            if guaranteed_profit_per_dollar > 1.0:
                guaranteed_profit_per_dollar = 0
                invalid_arb = True

            # Optimal allocation
            valid = [m for m in group if m.get("yes_price", 0) > 0]
            inv_prices = [1 / m["yes_price"] for m in valid]
            total_inv = sum(inv_prices)
            outcomes = []
            for m in valid:
                weight = (1 / m["yes_price"]) / total_inv if total_inv > 0 else 0
                outcomes.append({
                    "title": m["title"],
                    "price": m["yes_price"],
                    "probability": m["probability"],
                    "platform": m["platform"],
                    "url": m.get("url", ""),
                    "allocation_pct": round(weight * 100, 2),
                })

            results.append({
                "event_title": group[0]["title"].split("?")[0] + "?" if "?" in group[0]["title"] else group[0]["title"],
                "event_id": eid,
                "outcomes": outcomes,
                "num_outcomes": len(outcomes),
                "sum_probs": round(sum_probs, 4),
                "edge_pct": round(edge, 2),
                "guaranteed_profit_per_dollar": round(guaranteed_profit_per_dollar, 4),
                "invalid": invalid_arb,
            })

    results.sort(key=lambda x: x["edge_pct"], reverse=True)

    return results


# ── arb detection: tail-end ────────────────────────────────────────────────

@app.get("/arb/tail")
async def arb_tail(min_prob: float = Query(default=92)):
    data = await _fetch_all_markets()
    markets = data["markets"]

    results = []
    for m in markets:
        prob = m["probability"]
        if prob < min_prob:
            continue
        price = prob / 100
        if price >= 1:
            continue

        title = m["title"]
        vol = m.get("volume", 0) or 0

        # Filter bad tail-end titles: too short, generic patterns, low volume
        if len(title) < 15:
            continue
        # Check if the standalone contract name (after " — ") is a bad pattern
        parts = title.split(" — ")
        if len(parts) > 1 and BAD_TAIL_PATTERNS.match(parts[-1].strip()):
            continue
        # PredictIt reports 0 volume for all markets (broken API field), exempt them
        if vol < 1000 and m["platform"] != "predict":
            continue

        return_pct = (1 - price) / price * 100
        ct = parse_dt(m.get("close_time"))
        dtr = days_until(ct)
        d = max(dtr, 1) if dtr is not None else 90  # assume 90 days if unknown
        ann = return_pct / d * 365

        if ann <= 5:
            continue

        results.append({
            "id": m["id"],
            "title": title,
            "platform": m["platform"],
            "probability": prob,
            "price": round(price, 4),
            "return_pct": round(return_pct, 2),
            "days_to_resolution": dtr,
            "annualized_return": round(ann, 2),
            "url": m.get("url", ""),
            "volume": vol,
        })

    results.sort(key=lambda x: x["annualized_return"], reverse=True)
    return results


# ── arb detection: logical ─────────────────────────────────────────────────

SUBSET_PATTERNS = [
    # Sports team → conference
    (["chiefs", "patriots", "bills", "ravens", "bengals",
      "broncos", "raiders", "chargers", "dolphins", "jets",
      "steelers", "browns", "colts", "titans", "jaguars",
      "texans"], ["afc"]),
    (["eagles", "cowboys", "giants", "commanders", "49ers",
      "rams", "seahawks", "cardinals", "packers", "bears",
      "lions", "vikings", "saints", "falcons", "panthers",
      "buccaneers"], ["nfc"]),
    # Political subset
    (["trump", "desantis", "haley", "pence"], ["republican"]),
    (["biden", "harris", "newsom", "buttigieg"], ["democrat"]),
    # Crypto subset
    (["bitcoin", "ethereum", "solana", "bnb"],
     ["crypto", "cryptocurrency"]),
]


def _detect_subset(title_a: str, title_b: str) -> str | None:
    """If A is a subset of B, return explanation. Otherwise None.
    Both markets must be about the same event (keyword similarity > 0.25)."""
    # Require topic similarity to avoid spurious matches
    if keyword_similarity(title_a, title_b) < 0.15:
        return None
    a_low = title_a.lower()
    b_low = title_b.lower()
    for subset_kws, parent_kws in SUBSET_PATTERNS:
        a_has_subset = any(kw in a_low for kw in subset_kws)
        b_has_parent = any(kw in b_low for kw in parent_kws)
        if a_has_subset and b_has_parent:
            sub_match = next(kw for kw in subset_kws if kw in a_low)
            par_match = next(kw for kw in parent_kws if kw in b_low)
            return f"{sub_match.title()} is a subset of {par_match.upper()}"
        # Check reverse: B subset of A
        b_has_subset = any(kw in b_low for kw in subset_kws)
        a_has_parent = any(kw in a_low for kw in parent_kws)
        if b_has_subset and a_has_parent:
            return None  # signal caller to swap
    return None


def _detect_complement(title_a: str, title_b: str) -> bool:
    """Check if two markets look like complements of the same event."""
    sim = keyword_similarity(title_a, title_b)
    if sim < 0.4:
        return False
    a_low = title_a.lower()
    b_low = title_b.lower()
    complement_signals = [
        ("yes" in a_low and "no" in b_low) or ("no" in a_low and "yes" in b_low),
        ("over" in a_low and "under" in b_low) or ("under" in a_low and "over" in b_low),
        ("above" in a_low and "below" in b_low) or ("below" in a_low and "above" in b_low),
        ("win" in a_low and "lose" in b_low) or ("lose" in a_low and "win" in b_low),
    ]
    return any(complement_signals)


@app.get("/arb/logical")
async def arb_logical():
    data = await _fetch_all_markets()
    markets = data["markets"]

    opportunities = []
    seen = set()

    # Rule 1 — Subset rule
    for i, a in enumerate(markets):
        for j, b in enumerate(markets):
            if i >= j:
                continue
            if a["platform"] == b["platform"]:
                continue
            pair_key = tuple(sorted([a["id"] + a["platform"], b["id"] + b["platform"]]))
            if pair_key in seen:
                continue

            explanation = _detect_subset(a["title"], b["title"])
            if explanation is not None:
                # a is subset of b → prob(a) must be <= prob(b)
                if a["probability"] > b["probability"]:
                    gap = round(a["probability"] - b["probability"], 2)
                    seen.add(pair_key)
                    opportunities.append({
                        "title_a": a["title"],
                        "title_b": b["title"],
                        "prob_a": a["probability"],
                        "prob_b": b["probability"],
                        "platform_a": a["platform"],
                        "platform_b": b["platform"],
                        "logical_error": f"{explanation} — {a['probability']}% > {b['probability']}% is impossible",
                        "gap": gap,
                        "url_a": a.get("url", ""),
                        "url_b": b.get("url", ""),
                        "trade": f"BUY {b['title'][:60]} (should be >= {a['title'][:40]})",
                        "guaranteed": True,
                        "rule": "subset",
                    })
                continue

            # Check reverse: b subset of a
            explanation_rev = _detect_subset(b["title"], a["title"])
            if explanation_rev is not None:
                if b["probability"] > a["probability"]:
                    gap = round(b["probability"] - a["probability"], 2)
                    seen.add(pair_key)
                    opportunities.append({
                        "title_a": b["title"],
                        "title_b": a["title"],
                        "prob_a": b["probability"],
                        "prob_b": a["probability"],
                        "platform_a": b["platform"],
                        "platform_b": a["platform"],
                        "logical_error": f"{explanation_rev} — {b['probability']}% > {a['probability']}% is impossible",
                        "gap": gap,
                        "url_a": b.get("url", ""),
                        "url_b": a.get("url", ""),
                        "trade": f"BUY {a['title'][:60]} (should be >= {b['title'][:40]})",
                        "guaranteed": True,
                        "rule": "subset",
                    })

    # Rule 2 — Complement rule
    for i in range(len(markets)):
        for j in range(i + 1, len(markets)):
            a, b = markets[i], markets[j]
            if a["platform"] == b["platform"]:
                continue
            pair_key = tuple(sorted([a["id"] + a["platform"], b["id"] + b["platform"]]))
            if pair_key in seen:
                continue
            if not _detect_complement(a["title"], b["title"]):
                continue
            total = a["probability"] + b["probability"]
            if total < 93 or total > 107:
                gap = round(abs(total - 100), 2)
                seen.add(pair_key)
                if total < 93:
                    error = f"Complements sum to {total:.1f}% (should be ~100%) — {gap:.1f}pp under"
                    trade = f"BUY both YES sides — guaranteed {gap:.1f}pp edge"
                else:
                    error = f"Complements sum to {total:.1f}% (should be ~100%) — {gap:.1f}pp over"
                    trade = f"BUY both NO sides — guaranteed {gap:.1f}pp edge"
                opportunities.append({
                    "title_a": a["title"],
                    "title_b": b["title"],
                    "prob_a": a["probability"],
                    "prob_b": b["probability"],
                    "platform_a": a["platform"],
                    "platform_b": b["platform"],
                    "logical_error": error,
                    "gap": gap,
                    "url_a": a.get("url", ""),
                    "url_b": b.get("url", ""),
                    "trade": trade,
                    "guaranteed": True,
                    "rule": "complement",
                })

    opportunities.sort(key=lambda x: x["gap"], reverse=True)
    return {"opportunities": opportunities, "count": len(opportunities)}


# ── fast markets & fast arbs ──────────────────────────────────────────────

@app.get("/markets/fast")
async def markets_fast(max_days: int = Query(default=7)):
    data = await _fetch_all_markets()
    now = datetime.now(timezone.utc)

    results = []
    for m in data["markets"]:
        ct = parse_dt(m.get("close_time"))
        if ct is None:
            continue
        days_left = (ct - now).days
        if days_left < 0 or days_left > max_days:
            continue

        hours_left = days_left * 24
        if days_left == 0:
            urgency = "TODAY"
        elif days_left == 1:
            urgency = "TOMORROW"
        elif days_left <= 3:
            urgency = "3 DAYS"
        else:
            urgency = "THIS WEEK"

        results.append({
            **m,
            "days_left": days_left,
            "hours_left": hours_left,
            "urgency": urgency,
        })

    results.sort(key=lambda x: x["days_left"])
    return {"markets": results, "count": len(results), "max_days": max_days}


@app.get("/arb/fast")
async def arb_fast(max_days: int = Query(default=3), min_gap: float = Query(default=1.0)):
    cross = await arb_cross(min_gap=min_gap, min_volume=0)
    now = datetime.now(timezone.utc)

    results = []
    for arb in cross:
        # Check both sides resolve within max_days
        ct_high = parse_dt(arb.get("url_high") and None)  # no close_time in arb directly
        dtr = arb.get("days_to_resolution")
        if dtr is None or dtr > max_days:
            continue

        gap = arb["fee_adjusted_gap"]
        days = max(dtr, 0.04)
        ann = (gap / 100) / days * 365 * 100

        hours_left = round(dtr * 24, 1) if dtr < 2 else None
        if dtr < 1:
            urgency = "TODAY"
        elif dtr < 2:
            urgency = "TOMORROW"
        elif dtr <= 3:
            urgency = "THIS_WEEK"
        else:
            urgency = "SOON"

        results.append({
            **arb,
            "annualized_return": round(ann, 2),
            "hours_left": hours_left,
            "urgency": urgency,
        })

    results.sort(key=lambda x: x["annualized_return"], reverse=True)
    return results


# ── arb detection: combinatorial ───────────────────────────────────────────

TOPIC_KEYWORDS = {
    "us_president": ["president", "presidential", "white house", "potus", "election 2024", "election 2028"],
    "fed": ["federal reserve", "fed", "interest rate", "fomc", "rate cut", "rate hike"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto"],
    "ai": ["artificial intelligence", "openai", "chatgpt", "gpt", "agi", "ai"],
    "climate": ["climate", "temperature", "carbon", "emissions", "paris agreement"],
    "ukraine": ["ukraine", "russia", "crimea", "kyiv", "zelensky", "putin"],
    "china": ["china", "taiwan", "xi jinping", "ccp", "beijing"],
}


def categorize_market(title: str) -> list[str]:
    t = title.lower()
    cats = []
    for cat, keywords in TOPIC_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            cats.append(cat)
    return cats


@app.get("/arb/combinatorial")
async def arb_combinatorial(threshold: float = Query(default=15)):
    data = await _fetch_all_markets()
    markets = data["markets"]

    # Group by topic
    by_topic: dict[str, list[dict]] = {}
    for m in markets:
        cats = categorize_market(m["title"])
        for c in cats:
            by_topic.setdefault(c, []).append(m)

    flagged = []
    seen = set()
    for topic, group in by_topic.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                sim = title_similarity(a["title"], b["title"])
                if sim < 0.15:
                    continue
                gap = abs(a["probability"] - b["probability"])
                if gap < threshold:
                    continue
                pair_key = tuple(sorted([a["id"] + a["platform"], b["id"] + b["platform"]]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                flagged.append({
                    "topic": topic,
                    "market_a": {"title": a["title"], "probability": a["probability"],
                                 "platform": a["platform"], "url": a.get("url", "")},
                    "market_b": {"title": b["title"], "probability": b["probability"],
                                 "platform": b["platform"], "url": b.get("url", "")},
                    "gap": round(gap, 2),
                    "similarity": round(sim, 3),
                    "explanation": (
                        f"Both markets relate to '{topic}' but show a {gap:.0f}pp probability gap. "
                        f"If these events are logically linked, one may be mispriced."
                    ),
                })

    flagged.sort(key=lambda x: x["gap"], reverse=True)
    return flagged


# ── all arbs combined ──────────────────────────────────────────────────────

@app.get("/arb/all")
async def arb_all(
    min_gap: float = Query(default=1.5),
    min_volume: float = Query(default=0),
):
    cross_task = arb_cross(min_gap=min_gap, min_volume=min_volume)
    dutch_task = arb_dutch(min_edge=1)
    tail_task = arb_tail(min_prob=92)
    combo_task = arb_combinatorial(threshold=15)

    cross, dutch, tail, combo = await asyncio.gather(
        cross_task, dutch_task, tail_task, combo_task,
        return_exceptions=True,
    )

    cross = cross if isinstance(cross, list) else []
    dutch = dutch if isinstance(dutch, list) else []
    tail = tail if isinstance(tail, list) else []
    combo = combo if isinstance(combo, list) else []

    return {
        "cross": cross,
        "dutch": dutch,
        "tail": tail,
        "combinatorial": combo,
        "total_count": len(cross) + len(dutch) + len(tail) + len(combo),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── tradeability scorer ───────────────────────────────────────────────────

@app.post("/score")
async def score_opportunity(body: dict = Body(...)):
    plat_h = body.get("platform_high", "poly")
    plat_l = body.get("platform_low", "predict")
    prob_h = float(body.get("prob_high", 50))
    prob_l = float(body.get("prob_low", 50))
    gap = float(body.get("gap", abs(prob_h - prob_l)))
    vol_h = float(body.get("volume_high", 0))
    vol_l = float(body.get("volume_low", 0))
    dtr = body.get("days_to_resolution")
    title_a = body.get("title_a", "")
    title_b = body.get("title_b", "")

    fee_gap = gap - PLATFORM_FEES.get(plat_h, 0.02) * 100 - PLATFORM_FEES.get(plat_l, 0.02) * 100

    warnings = []

    # 1. Gap quality (0-25)
    if fee_gap < 0:
        return {
            "score": 0, "grade": "F", "tradeable": False,
            "breakdown": {"gap_score": 0, "liquidity_score": 0, "time_score": 0,
                          "platform_score": 0, "resolution_score": 0},
            "warnings": [f"Fee-adjusted gap is negative ({fee_gap:.1f}%). This is a losing trade."],
            "recommendation": "Do not take this trade. After platform fees, there is no edge.",
        }

    if fee_gap > 5:
        gap_score = 25
    elif fee_gap >= 2:
        gap_score = 15
    else:
        gap_score = 8
        warnings.append(f"Small fee-adjusted gap ({fee_gap:.1f}%) — tight margin for error.")

    # 2. Liquidity (0-20)
    min_vol = min(vol_h, vol_l)
    if min_vol > 10000:
        liq_score = 20
    elif min_vol > 2000:
        liq_score = 12
    elif min_vol > 500:
        liq_score = 5
    else:
        liq_score = 0
        warnings.append(f"Very low liquidity (${min_vol:.0f}) — may not be able to fill orders.")

    # 3. Time value (0-20)
    if dtr is not None:
        dtr = int(dtr)
        if dtr < 1:
            time_score = 15
            warnings.append("Resolves very soon — price can spike, be careful.")
        elif dtr <= 7:
            time_score = 20
        elif dtr <= 30:
            time_score = 18
        elif dtr <= 180:
            time_score = 10
        else:
            time_score = 3
            warnings.append(f"Capital locked for {dtr} days — high opportunity cost.")
    else:
        time_score = 8
        warnings.append("No resolution date available — unknown time horizon.")

    # 4. Platform reliability (0-15)
    high_vol_threshold = 1000
    h_reliable = vol_h > high_vol_threshold
    l_reliable = vol_l > high_vol_threshold
    if h_reliable and l_reliable:
        plat_score = 15
    elif h_reliable or l_reliable:
        plat_score = 8
    else:
        plat_score = 3
        warnings.append("Both platforms have low volume — questionable price accuracy.")

    # 5. Resolution risk (0-20)
    sim = title_similarity(title_a, title_b) if title_a and title_b else 0.5
    if sim > 0.8:
        res_score = 20
    elif sim > 0.6:
        res_score = 15
    elif sim >= 0.40:
        res_score = 8
        warnings.append(f"Moderate title overlap ({sim:.2f}) — verify both markets resolve the same event.")
    else:
        res_score = 3
        warnings.append(f"Low title overlap ({sim:.2f}) — high risk these are different events.")

    # Platform-specific warnings
    if plat_l == "predict" or plat_h == "predict":
        warnings.append("PredictIt charges 10% fee on profits + 5% withdrawal fee.")
    if plat_l == "poly" or plat_h == "poly":
        if fee_gap < 3:
            warnings.append("Polymarket 2% fee leaves very thin margin.")

    total = gap_score + liq_score + time_score + plat_score + res_score

    if total > 80:
        grade = "A"
    elif total > 60:
        grade = "B"
    elif total > 40:
        grade = "C"
    elif total > 20:
        grade = "D"
    else:
        grade = "F"

    if total >= 80:
        rec = "Strong opportunity. Good edge, liquidity, and resolution confidence. Consider sizing up to half-Kelly."
    elif total >= 60:
        rec = "Moderate opportunity. Check exact question wording on both platforms before committing capital."
    elif total >= 40:
        rec = "Marginal opportunity. Only trade with small size and close monitoring. Several risk factors present."
    else:
        rec = "Weak opportunity. High risk of losing money due to fees, liquidity, or resolution mismatch."

    return {
        "score": total,
        "grade": grade,
        "tradeable": total >= 60,
        "breakdown": {
            "gap_score": gap_score,
            "liquidity_score": liq_score,
            "time_score": time_score,
            "platform_score": plat_score,
            "resolution_score": res_score,
        },
        "warnings": warnings,
        "recommendation": rec,
    }


# ── order book depth ───────────────────────────────────────────────────────

@app.get("/market/{platform}/{market_id}/depth")
async def market_depth(platform: str, market_id: str):
    if platform != "poly":
        return {"error": "Order book depth only available for Polymarket"}

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                f"https://clob.polymarket.com/book?token_id={market_id}",
            )
            resp.raise_for_status()
            data = resp.json()

        bids = [{"price": float(b.get("price", 0)), "size": float(b.get("size", 0))}
                for b in (data.get("bids", []) or [])[:5]]
        asks = [{"price": float(a.get("price", 0)), "size": float(a.get("size", 0))}
                for a in (data.get("asks", []) or [])[:5]]

        best_bid = bids[0]["price"] if bids else 0
        best_ask = asks[0]["price"] if asks else 0

        return {
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": round(best_ask - best_bid, 4) if best_bid and best_ask else None,
            "total_bid_liquidity": round(sum(b["price"] * b["size"] for b in bids), 2),
        }
    except Exception as e:
        logger.error(f"[depth] {e}")
        return {"bids": [], "asks": [], "best_bid": 0, "best_ask": 0, "spread": None, "total_bid_liquidity": 0}


# ── history ─────────────────────────────────────────────────────────────────

@app.get("/history")
async def get_history():
    return load_history()


@app.post("/history/log")
async def log_history(body: dict = Body(...)):
    history = load_history()
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "taken": False,
        "result": None,
        "notes": "",
        **body,
    }
    history.append(entry)
    save_history(history)
    return entry


@app.patch("/history/{entry_id}")
async def update_history(entry_id: str, body: dict = Body(...)):
    history = load_history()
    for entry in history:
        if entry.get("id") == entry_id:
            entry.update(body)
            save_history(history)
            return entry
    return {"error": "not found"}


@app.delete("/history/{entry_id}")
async def delete_history(entry_id: str):
    history = load_history()
    before = len(history)
    history = [h for h in history if h.get("id") != entry_id]
    if len(history) < before:
        save_history(history)
        return {"ok": True}
    return {"error": "not found"}


# ── stats ───────────────────────────────────────────────────────────────────

@app.get("/stats")
async def get_stats():
    history = load_history()
    total = len(history)
    taken = [h for h in history if h.get("taken")]
    results = [h.get("result") for h in taken if h.get("result") is not None]
    wins = [r for r in results if isinstance(r, (int, float)) and r > 0]
    losses = [r for r in results if isinstance(r, (int, float)) and r <= 0]

    return {
        "total_logged": total,
        "total_taken": len(taken),
        "total_with_results": len(results),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": round(len(wins) / len(results) * 100, 1) if results else None,
        "avg_gap_taken": round(np.mean([h.get("gap", 0) or h.get("raw_gap", 0) for h in taken]), 2) if taken else None,
        "best_trade": max(results) if results else None,
        "worst_trade": min(results) if results else None,
    }


# ── health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    total_markets = sum(s.get("count", 0) for s in _platform_status.values())
    all_down = all(s.get("status") == "down" for s in _platform_status.values())

    return {
        "status": "degraded" if all_down else "ok",
        "platforms": dict(_platform_status),
        "total_markets": total_markets,
        "uptime_seconds": round(time.time() - START_TIME),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.0",
    }


# ── AI prediction engine ───────────────────────────────────────────────

AI_SYSTEM_PROMPT = (
    "You are a professional prediction market analyst with deep expertise "
    "in political forecasting, economics, and sports. Analyze the given "
    "market and provide a structured assessment. Be precise, data-driven, "
    "and honest about uncertainty. Never fabricate data."
)


@app.post("/ai/predict")
async def ai_predict(body: dict = Body(...)):
    title = body.get("title", "")
    platform = body.get("platform", "unknown")
    probability = float(body.get("probability", 50))
    volume = float(body.get("volume", 0))
    days = body.get("days_to_resolution")
    cache_key = f"ai_predict:{title}:{platform}"
    cached = _ai_cache.get(cache_key)
    if cached and time.time() < cached["expires_at"]:
        _req_metrics["cache_hits"] += 1
        return cached["data"]
    _req_metrics["cache_misses"] += 1
    if not _ai_client:
        return {"error": "Add ANTHROPIC_API_KEY to .env to enable AI analysis."}
    try:
        user_msg = (
            f"Analyze this prediction market:\nTitle: {title}\n"
            f"Current probability: {probability}%\nPlatform: {platform}\n"
            f"Volume: ${volume:,.0f}\nDays to resolution: {days or 'Unknown'}\n\n"
            f"Provide: 1. Fair value estimate 2. Key factors 3. Overpriced/underpriced/fair "
            f"4. Historical comps 5. Top 3 risks 6. Recommendation\n\n"
            f'Respond ONLY in JSON with keys: fair_value, confidence_low, confidence_high, '
            f'mispricing, mispricing_magnitude, key_factors, historical_comps, risks, '
            f'recommendation, reasoning, confidence_score'
        )
        message = _ai_client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024,
            system=AI_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = message.content[0].text
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', text)
            result = json.loads(match.group()) if match else {}
        _ai_cache[cache_key] = {"data": result, "expires_at": time.time() + 1800}
        return result
    except Exception as e:
        logger.error(f"[ai/predict] {e}")
        return {"error": f"AI analysis failed: {e}"}


@app.post("/ai/analyze-arb")
async def ai_analyze_arb(body: dict = Body(...)):
    prob_high = float(body.get("prob_high", 50))
    prob_low = float(body.get("prob_low", 50))
    gap = float(body.get("gap", abs(prob_high - prob_low)))
    vol_h = body.get("volume_high", 0) or 0
    vol_l = body.get("volume_low", 0) or 0
    plat_h = body.get("platform_high", "")
    plat_l = body.get("platform_low", "")
    if gap > 20:
        verdict, risk = "INVESTIGATE", "high"
    elif gap > 5:
        verdict, risk = "TAKE_ARB", "low"
    else:
        verdict, risk = "AVOID", "medium"
    return {
        "same_event": None,
        "confidence": None,
        "source": "heuristic",
        "price_explanation": f"The {gap:.1f}pp gap reflects different trader bases and liquidity.",
        "more_accurate_platform": plat_h if (vol_h or 0) > (vol_l or 0) else plat_l,
        "resolution_risk": risk,
        "resolution_risk_reason": f"{'Large gap suggests resolution risk' if gap > 20 else 'Reasonable alignment' if gap > 5 else 'Thin edge after costs'}",
        "arb_recommendation": f"{'Take with half-Kelly' if verdict == 'TAKE_ARB' else 'Investigate first' if verdict == 'INVESTIGATE' else 'Skip'}",
        "verdict": verdict,
    }


@app.post("/ai/chat")
async def ai_chat(body: dict = Body(...)):
    message_text = body.get("message", "")
    context = body.get("context", {})
    if not _ai_client:
        return {
            "response": "Add ANTHROPIC_API_KEY to .env to enable AI chat.",
            "error": True,
        }
    try:
        sys_prompt = (
            "You are ArbRadar AI, an expert prediction market analyst. "
            "Be direct, specific, and actionable.\n\n"
            f"Context: {json.dumps(context, default=str)[:500]}"
        )
        message = _ai_client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024,
            system=sys_prompt,
            messages=[{"role": "user", "content": message_text}],
        )
        return {"response": message.content[0].text}
    except Exception as e:
        return {"response": f"Error: {e}", "error": True}


@app.post("/ai/portfolio")
async def ai_portfolio(body: dict = Body(...)):
    positions = body.get("positions", [])
    bankroll = float(body.get("bankroll", 500))
    total_invested = sum(p.get("amount", 0) for p in positions)
    return {
        "source": "heuristic",
        "correlation_risk": "low" if len(positions) < 3 else "medium",
        "total_exposure_pct": round(total_invested / max(bankroll, 1) * 100, 1),
        "diversification_score": min(100, len(positions) * 20),
        "suggestions": [
            "Diversify across more platforms to reduce correlation risk",
            "Keep total exposure below 50% of bankroll",
            "Consider hedging correlated positions",
        ],
        "summary": f"{len(positions)} positions, ${total_invested:.0f} invested of ${bankroll:.0f} bankroll.",
    }


# ── news & sentiment ──────────────────────────────────────────────────

@app.get("/news/{query}")
async def get_news(query: str):
    return {"query": query, "articles": [], "error": "News feed not configured. No live news source available."}


@app.get("/sentiment/{title}")
async def get_sentiment(title: str):
    return {"error": "Sentiment analysis not configured. No live sentiment source available."}


# ── analytics ──────────────────────────────────────────────────────────

@app.get("/analytics")
async def get_analytics():
    history = load_history()
    taken = [h for h in history if h.get("taken")]
    results_list = [h for h in taken if h.get("result") is not None]
    results_vals = [float(h["result"]) for h in results_list]
    wins = [r for r in results_vals if r > 0]
    plat_breakdown = {}
    for h in taken:
        p = h.get("platform_high", h.get("platform", "unknown"))
        if p not in plat_breakdown:
            plat_breakdown[p] = {"taken": 0, "won": 0, "pnl": 0}
        plat_breakdown[p]["taken"] += 1
        if h.get("result") is not None:
            plat_breakdown[p]["pnl"] += float(h["result"])
            if float(h["result"]) > 0:
                plat_breakdown[p]["won"] += 1
    monthly = {}
    for h in results_list:
        month = (h.get("timestamp") or "")[:7]
        if month:
            if month not in monthly:
                monthly[month] = {"month": month, "pnl": 0, "trades": 0}
            monthly[month]["trades"] += 1
            monthly[month]["pnl"] += float(h["result"])
    total_pnl = sum(results_vals) if results_vals else 0
    bank = load_settings().get("bankroll", 500)
    for m in monthly.values():
        m["return_pct"] = round(m["pnl"] / max(bank, 1) * 100, 2)
    return {
        "total_opportunities_logged": len(history),
        "total_taken": len(taken),
        "win_rate": round(len(wins) / max(len(results_vals), 1) * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_gap_taken": round(float(np.mean([h.get("raw_gap", 0) or 0 for h in taken])), 2) if taken else 0,
        "platform_breakdown": plat_breakdown,
        "monthly_returns": sorted(monthly.values(), key=lambda x: x["month"]),
        "sharpe_ratio": 0,
        "best_trade": max(results_vals) if results_vals else 0,
        "worst_trade": min(results_vals) if results_vals else 0,
    }


# ── watchlist ──────────────────────────────────────────────────────────

@app.get("/watchlist")
async def get_watchlist():
    return load_watchlist()


@app.post("/watchlist")
async def add_watchlist(body: dict = Body(...)):
    wl = load_watchlist()
    entry = {
        "id": body.get("id", str(uuid.uuid4())[:8]),
        "title": body.get("title", ""),
        "platform": body.get("platform", ""),
        "probability": body.get("probability", 0),
        "volume": body.get("volume", 0),
        "url": body.get("url", ""),
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    wl.append(entry)
    save_watchlist(wl)
    return entry


@app.delete("/watchlist/{item_id}")
async def remove_watchlist(item_id: str):
    wl = load_watchlist()
    before = len(wl)
    wl = [w for w in wl if w.get("id") != item_id]
    if len(wl) < before:
        save_watchlist(wl)
        return {"ok": True}
    return {"error": "not found"}


# ── settings ───────────────────────────────────────────────────────────

@app.get("/settings")
async def get_app_settings():
    return load_settings()


@app.post("/settings")
async def update_app_settings(body: dict = Body(...)):
    current = load_settings()
    current.update(body)
    save_settings(current)
    return current


# ── metrics ────────────────────────────────────────────────────────────

@app.get("/metrics")
async def get_app_metrics():
    now = time.time()
    recent = [t for t in _req_metrics["errors"] if now - t < 3600]
    _req_metrics["errors"] = recent
    return {
        "uptime_seconds": round(now - START_TIME),
        "total_requests": _req_metrics["total"],
        "cache_hits": _req_metrics["cache_hits"],
        "cache_misses": _req_metrics["cache_misses"],
        "errors_last_hour": len(recent),
        "version": "3.0",
    }


@app.middleware("http")
async def count_requests(request, call_next):
    _req_metrics["total"] += 1
    try:
        resp = await call_next(request)
        if resp.status_code >= 500:
            _req_metrics["errors"].append(time.time())
        return resp
    except Exception:
        _req_metrics["errors"].append(time.time())
        raise


# ── serve frontend ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse((BASE / "index.html").read_text())
