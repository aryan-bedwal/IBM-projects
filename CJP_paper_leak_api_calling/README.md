# News Scraper — GNews API (Cockroach Janta Party Protest Coverage)

Calls the [GNews API](https://gnews.io/) to pull news articles for a topic,
stores them in a local SQLite database (deduplicated by URL), and gives you
simple trend/source analysis on top.

## Files
- `config.py` — API key, search queries, and settings. Edit `SEARCH_QUERIES` here.
- `news_scraper.py` — calls the GNews API, saves new articles to `news.db`, skips duplicates.
- `analyze.py` — prints article counts over time, top sources, and latest headlines.

## Setup

```bash
pip install requests pandas python-dotenv
```

Get a **free** API key at https://gnews.io/ (100 requests/day on the free tier), then:

```bash
export GNEWS_API_KEY="your_key_here"
```

## Usage

Run with the default queries from `config.py`:
```bash
python news_scraper.py
```

Run with a custom query and export to CSV:
```bash
python news_scraper.py --query "Cockroach Janta Party protest" --max-results 10 --export-csv
```

View trends and latest headlines:
```bash
python analyze.py
```

## Keeping it running automatically

To poll for new articles periodically, add a cron job (Linux/Mac):
```bash
# every hour
0 * * * * cd /path/to/news_scraper && /usr/bin/python3 news_scraper.py >> scraper.log 2>&1
```

## Notes
- Deduplication is by article URL — running the scraper repeatedly is safe;
  it will only insert new articles.
- Free-tier GNews limits results per request to 10 and total requests to
  100/day — space out runs (e.g. hourly) to stay within quota.
- Edit `SEARCH_QUERIES` in `config.py` as the story's terminology evolves
  (e.g. add hashtags, alternate spellings, or related keywords).
