"""
Analyze scraped articles: article counts over time, top sources, latest headlines.

Run:
    python analyze.py
"""

import sqlite3

import pandas as pd

import config


def load_data(db_path: str = config.DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    conn.close()
    if df.empty:
        return df
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["published_date"] = df["published_at"].dt.date
    return df


def main():
    df = load_data()
    if df.empty:
        print("No articles found yet. Run news_scraper.py first.")
        return

    print(f"Total articles stored: {len(df)}\n")

    print("Articles per day:")
    print(df.groupby("published_date").size().sort_index())

    print("\nTop sources:")
    print(df["source_name"].value_counts().head(10))

    print("\nLatest 10 headlines:")
    latest = df.sort_values("published_at", ascending=False).head(10)
    for _, row in latest.iterrows():
        date_str = row["published_at"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["published_at"]) else "unknown"
        print(f"  [{date_str}] {row['title']}  ({row['source_name']})")
        print(f"      {row['url']}")


if __name__ == "__main__":
    main()
