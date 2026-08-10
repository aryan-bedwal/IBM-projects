"""
Configuration for the news scraper.
Set your GNews API key as an environment variable (recommended) or paste it below.

Get a free API key at: https://gnews.io/ (free tier: 100 requests/day)
"""

import os

# Prefer environment variable; fall back to placeholder for quick local testing.
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "YOUR_GNEWS_API_KEY_HERE")

# Base search terms — edit/add variations as the story develops.
SEARCH_QUERIES = [
    "Cockroach Janta Party protest",
    "Cockroach Janta Party",
]

# GNews search params
LANGUAGE = "en"
COUNTRY = "in"          # India
MAX_RESULTS_PER_QUERY = 10   # free tier max per request
SORT_BY = "publishedAt"      # "publishedAt" or "relevance"

# Storage
DB_PATH = "news.db"
CSV_EXPORT_PATH = "news_export.csv"
