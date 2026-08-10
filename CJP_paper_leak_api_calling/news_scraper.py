"""
News Scraper — GNews API
=========================
Calls the GNews API for configured search queries, deduplicates against
previously stored articles, and saves new ones to a local SQLite database
(plus an optional CSV export).

Setup:
    export GNEWS_API_KEY="your_key_here"   # get one free at https://gnews.io/
    pip install requests pandas python-dotenv

Run:
    python news_scraper.py
    python news_scraper.py --query "Cockroach Janta Party" --max-results 10
    python news_scraper.py --export-csv
"""

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone

import requests

import config

GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            description TEXT,
            content TEXT,
            source_name TEXT,
            published_at TEXT,
            query TEXT,
            scraped_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def fetch_articles(query: str, api_key: str, max_results: int = 10,
                    language: str = "en", country: str = "in",
                    sortby: str = "publishedAt") -> list[dict]:
    """Call the GNews /search endpoint for a single query. Returns list of article dicts."""
    params = {
        "q": query,
        "lang": language,
        "country": country,
        "max": max_results,
        "sortby": sortby,
        "apikey": api_key,
    }
    resp = requests.get(GNEWS_SEARCH_URL, params=params, timeout=15)

    if resp.status_code == 401:
        raise RuntimeError("GNews API rejected the key (401). Check GNEWS_API_KEY.")
    if resp.status_code == 403:
        raise RuntimeError("GNews API quota exceeded or forbidden (403).")
    resp.raise_for_status()

    data = resp.json()
    return data.get("articles", [])


def save_articles(conn: sqlite3.Connection, articles: list[dict], query: str) -> int:
    """Insert new articles, skipping duplicates by URL. Returns count of new rows inserted."""
    inserted = 0
    now = datetime.now(timezone.utc).isoformat()
    for a in articles:
        try:
            conn.execute(
                """
                INSERT INTO articles (url, title, description, content, source_name,
                                       published_at, query, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a.get("url"),
                    a.get("title"),
                    a.get("description"),
                    a.get("content"),
                    (a.get("source") or {}).get("name"),
                    a.get("publishedAt"),
                    query,
                    now,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            # URL already exists — duplicate, skip
            continue
    conn.commit()
    return inserted


def export_csv(conn: sqlite3.Connection, csv_path: str):
    import pandas as pd

    df = pd.read_sql_query("SELECT * FROM articles ORDER BY published_at DESC", conn)
    df.to_csv(csv_path, index=False)
    print(f"Exported {len(df)} articles to {csv_path}")


def run(queries: list[str], max_results: int, export_csv_flag: bool):
    if config.GNEWS_API_KEY == "YOUR_GNEWS_API_KEY_HERE":
        print(
            "ERROR: No API key set. Run:\n"
            "  export GNEWS_API_KEY='your_key_here'\n"
            "Get a free key at https://gnews.io/",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = init_db(config.DB_PATH)
    total_new = 0

    for q in queries:
        print(f"Searching: '{q}' ...")
        try:
            articles = fetch_articles(
                query=q,
                api_key=config.GNEWS_API_KEY,
                max_results=max_results,
                language=config.LANGUAGE,
                country=config.COUNTRY,
                sortby=config.SORT_BY,
            )
        except Exception as e:
            print(f"  Failed: {e}", file=sys.stderr)
            continue

        new_count = save_articles(conn, articles, q)
        total_new += new_count
        print(f"  Found {len(articles)} articles, {new_count} new.")
        time.sleep(1)  # be polite to the API / respect rate limits

    print(f"\nDone. {total_new} new articles saved to {config.DB_PATH}")

    if export_csv_flag:
        export_csv(conn, config.CSV_EXPORT_PATH)

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Scrape news via GNews API")
    parser.add_argument("--query", action="append", help="Search query (repeatable). Defaults to config.SEARCH_QUERIES")
    parser.add_argument("--max-results", type=int, default=config.MAX_RESULTS_PER_QUERY)
    parser.add_argument("--export-csv", action="store_true", help="Also export all stored articles to CSV")
    args = parser.parse_args()

    queries = args.query if args.query else config.SEARCH_QUERIES
    run(queries, args.max_results, args.export_csv)


if __name__ == "__main__":
    main()
