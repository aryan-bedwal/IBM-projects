"""
CineMatch — Content-Based Movie Recommendation Dashboard
Run with:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Page config + theme
# --------------------------------------------------------------------------
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

GOLD = "#E7B24B"
TEAL = "#4FB8A6"
ROSE = "#D1667A"
PERIWINKLE = "#7C86D6"
SURFACE = "#1D1922"
RULE = "#332C3B"
TEXT_MUTED = "#A79FAE"

GENRE_THEME = {
    "Sci-Fi": ("🚀", f"linear-gradient(150deg, {PERIWINKLE}, #2B2F63)"),
    "Thriller": ("🔍", "linear-gradient(150deg, #8B4B8F, #3B2049)"),
    "Action": ("⚡", f"linear-gradient(150deg, {ROSE}, #7A2E42)"),
    "Crime": ("🏆", "linear-gradient(150deg, #4B4E63, #16171F)"),
    "Drama": ("🎭", f"linear-gradient(150deg, {TEAL}, #1F5F58)"),
    "Comedy": ("😂", f"linear-gradient(150deg, {GOLD}, #8A5A1E)"),
    "Romance": ("❤️", f"linear-gradient(150deg, {ROSE}, #8A3A52)"),
    "Horror": ("👻", "linear-gradient(150deg, #3E3F4E, #0C0C13)"),
    "Animation": ("✨", f"linear-gradient(150deg, {GOLD}, {TEAL})"),
    "Fantasy": ("✨", f"linear-gradient(150deg, {PERIWINKLE}, #8B4B8F)"),
    "Mystery": ("❓", "linear-gradient(150deg, #5C4B8F, #241C3B)"),
}

st.markdown(
    f"""
    <style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{padding-top: 1.6rem; max-width: 1200px;}}
    .cm-title {{
        font-size: 46px; font-weight: 800; letter-spacing: 2px;
        color: {GOLD}; margin-bottom: 0; line-height: 1;
    }}
    .cm-tagline {{ color: {TEXT_MUTED}; font-size: 14px; margin-top: 4px; }}
    .cm-sprockets {{
        display: flex; gap: 8px; margin: 14px 0 22px;
    }}
    .cm-sprockets span {{
        width: 5px; height: 5px; border-radius: 50%; background: {RULE}; display: inline-block;
    }}
    .cm-card {{
        background: {SURFACE}; border: 1px solid {RULE}; border-radius: 10px;
        padding: 10px; margin-bottom: 12px;
    }}
    .cm-poster {{
        height: 90px; border-radius: 6px; display: flex; align-items: center;
        justify-content: center; font-size: 30px; margin-bottom: 8px; position: relative;
    }}
    .cm-year {{
        position: absolute; bottom: 4px; right: 8px; font-size: 10px;
        color: rgba(255,255,255,0.75); font-family: monospace;
    }}
    .cm-badge {{
        display: inline-block; font-size: 10.5px; color: {GOLD}; border: 1px solid #8A6A2E;
        border-radius: 4px; padding: 1px 6px; margin: 2px 3px 0 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    movies_data = [
        {"title": "Inception", "year": 2010, "genres": ["Sci-Fi", "Thriller"], "tags": ["dreams", "heist", "time-loop"], "rating": 8.8},
        {"title": "The Dark Knight", "year": 2008, "genres": ["Action", "Crime", "Thriller"], "tags": ["vigilante", "moral-dilemma", "urban"], "rating": 9.0},
        {"title": "Interstellar", "year": 2014, "genres": ["Sci-Fi", "Drama"], "tags": ["space", "time-dilation", "family"], "rating": 8.7},
        {"title": "The Grand Budapest Hotel", "year": 2014, "genres": ["Comedy", "Drama"], "tags": ["ensemble", "whimsical", "road-trip"], "rating": 8.1},
        {"title": "Parasite", "year": 2019, "genres": ["Thriller", "Drama"], "tags": ["class-divide", "dark-comedy", "twist"], "rating": 8.6},
        {"title": "La La Land", "year": 2016, "genres": ["Romance", "Drama"], "tags": ["musical", "dream-chasing", "bittersweet"], "rating": 8.0},
        {"title": "Get Out", "year": 2017, "genres": ["Horror", "Thriller"], "tags": ["social-commentary", "twist", "paranoia"], "rating": 7.7},
        {"title": "Spirited Away", "year": 2001, "genres": ["Animation", "Fantasy"], "tags": ["coming-of-age", "spirits", "journey"], "rating": 8.6},
        {"title": "Mad Max: Fury Road", "year": 2015, "genres": ["Action", "Sci-Fi"], "tags": ["chase", "dystopia", "survival"], "rating": 8.1},
        {"title": "Whiplash", "year": 2014, "genres": ["Drama"], "tags": ["mentorship", "obsession", "jazz-drumming"], "rating": 8.5},
        {"title": "Knives Out", "year": 2019, "genres": ["Mystery", "Comedy"], "tags": ["whodunit", "family", "twist"], "rating": 7.9},
        {"title": "Coco", "year": 2017, "genres": ["Animation", "Fantasy"], "tags": ["family", "music", "afterlife"], "rating": 8.4},
        {"title": "John Wick", "year": 2014, "genres": ["Action", "Thriller"], "tags": ["revenge", "assassin", "stylized"], "rating": 7.4},
        {"title": "Her", "year": 2013, "genres": ["Romance", "Sci-Fi"], "tags": ["AI", "loneliness", "near-future"], "rating": 8.0},
        {"title": "The Conjuring", "year": 2013, "genres": ["Horror"], "tags": ["haunting", "supernatural", "investigation"], "rating": 7.5},
        {"title": "Big Hero 6", "year": 2014, "genres": ["Animation", "Action"], "tags": ["friendship", "robot", "heroism"], "rating": 7.8},
        {"title": "Se7en", "year": 1995, "genres": ["Crime", "Thriller"], "tags": ["serial-killer", "detective", "dark"], "rating": 8.6},
        {"title": "The Shawshank Redemption", "year": 1994, "genres": ["Drama"], "tags": ["hope", "friendship", "prison"], "rating": 9.3},
        {"title": "Pulp Fiction", "year": 1994, "genres": ["Crime", "Drama"], "tags": ["nonlinear", "dialogue-driven", "violence"], "rating": 8.9},
        {"title": "Titanic", "year": 1997, "genres": ["Romance", "Drama"], "tags": ["disaster", "forbidden-love", "epic"], "rating": 7.9},
        {"title": "The Hangover", "year": 2009, "genres": ["Comedy"], "tags": ["bachelor-party", "chaos", "friendship"], "rating": 7.7},
        {"title": "Superbad", "year": 2007, "genres": ["Comedy"], "tags": ["coming-of-age", "friendship", "teen"], "rating": 7.6},
        {"title": "Arrival", "year": 2016, "genres": ["Sci-Fi", "Drama"], "tags": ["first-contact", "linguistics", "time"], "rating": 7.9},
        {"title": "Edge of Tomorrow", "year": 2014, "genres": ["Sci-Fi", "Action"], "tags": ["time-loop", "alien-invasion", "soldier"], "rating": 7.9},
        {"title": "The Prestige", "year": 2006, "genres": ["Drama", "Mystery", "Thriller"], "tags": ["rivalry", "magic", "twist"], "rating": 8.5},
        {"title": "Toy Story", "year": 1995, "genres": ["Animation", "Comedy"], "tags": ["friendship", "toys", "adventure"], "rating": 8.3},
        {"title": "Zootopia", "year": 2016, "genres": ["Animation", "Comedy", "Mystery"], "tags": ["social-commentary", "buddy", "investigation"], "rating": 8.0},
        {"title": "A Quiet Place", "year": 2018, "genres": ["Horror"], "tags": ["survival", "silence", "family"], "rating": 7.5},
    ]
    df = pd.DataFrame(movies_data)
    df["movie_id"] = df.index
    return df


@st.cache_resource
def build_engine(df: pd.DataFrame):
    soup = df.apply(lambda r: " ".join(r["genres"] * 2 + r["tags"]), axis=1)
    vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
    tfidf_matrix = vectorizer.fit_transform(soup)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix


def recommend(df, similarity_matrix, title, top_n=6):
    idx = df.index[df["title"] == title][0]
    scores = [(i, s) for i, s in enumerate(similarity_matrix[idx]) if i != idx]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_n]
    seed = df.loc[idx]

    results = []
    for i, score in top:
        row = df.loc[i]
        shared_genres = sorted(set(seed["genres"]) & set(row["genres"]))
        shared_tags = sorted(set(seed["tags"]) & set(row["tags"]))
        results.append({
            "title": row["title"], "year": row["year"], "rating": row["rating"],
            "genres": row["genres"], "match_score": round(score * 100, 1),
            "shared_genres": shared_genres, "shared_tags": shared_tags,
        })
    return results


def poster_html(movie_genres, year, size=90, emoji_size=28):
    icon, grad = GENRE_THEME.get(movie_genres[0], ("🎬", f"linear-gradient(150deg, {TEAL}, #1F5F58)"))
    return (
        f'<div class="cm-poster" style="background:{grad}; height:{size}px; font-size:{emoji_size}px;">'
        f'{icon}<span class="cm-year">{year}</span></div>'
    )


def badges_html(items, limit=3):
    return "".join(f'<span class="cm-badge">{g}</span>' for g in items[:limit])


# --------------------------------------------------------------------------
# Load + build
# --------------------------------------------------------------------------
df = load_data()
similarity_matrix = build_engine(df)
all_genres = sorted({g for genres in df["genres"] for g in genres})

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🎬 CineMatch")
    st.caption("Content-based recommender · TF-IDF + cosine similarity")
    st.divider()

    search_query = st.text_input("Search titles", "")
    genre_filter = st.multiselect("Filter by genre", all_genres)

    st.divider()
    st.markdown("**Get recommendations**")
    seed_title = st.selectbox("Pick a movie you liked", sorted(df["title"].tolist()))
    top_n = st.slider("Number of recommendations", 3, 10, 6)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="cm-title">CINEMATCH</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cm-tagline">Pick a title you like — matches are found on shared genres and themes, no viewing history required.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="cm-sprockets">' + "<span></span>" * 40 + "</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Stats row
# --------------------------------------------------------------------------
genre_counts = pd.Series([g for genres in df["genres"] for g in genres]).value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Titles Tracked", len(df))
c2.metric("Avg Rating", f"{df['rating'].mean():.1f}")
c3.metric("Genres", len(all_genres))
c4.metric("Leading Genre", genre_counts.idxmax())

st.write("")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_browse, tab_recs, tab_analytics = st.tabs(["🎬 Browse", "🎯 Recommendations", "📊 Analytics"])

# --- Browse tab -----------------------------------------------------------
with tab_browse:
    filtered = df[df["title"].str.contains(search_query, case=False, na=False)]
    if genre_filter:
        filtered = filtered[filtered["genres"].apply(lambda gs: any(g in gs for g in genre_filter))]

    if filtered.empty:
        st.info("No titles match that search — try a different genre or keyword.")
    else:
        cols = st.columns(5)
        for i, (_, m) in enumerate(filtered.iterrows()):
            with cols[i % 5]:
                st.markdown(
                    f'<div class="cm-card">{poster_html(m["genres"], m["year"])}'
                    f'<div style="font-weight:700; font-size:13px;">{m["title"]}</div>'
                    f'<div style="color:{TEXT_MUTED}; font-size:11.5px; margin-top:2px;">⭐ {m["rating"]}</div>'
                    f'<div>{badges_html(m["genres"])}</div></div>',
                    unsafe_allow_html=True,
                )

# --- Recommendations tab ---------------------------------------------------
with tab_recs:
    seed = df[df["title"] == seed_title].iloc[0]
    left, right = st.columns([1, 2])

    with left:
        st.markdown(
            f'<div class="cm-card">{poster_html(seed["genres"], seed["year"], size=180, emoji_size=54)}'
            f'<div style="font-weight:700; font-size:17px;">{seed["title"]}</div>'
            f'<div style="color:{TEXT_MUTED}; font-size:12.5px; margin-top:2px;">{seed["year"]} · ⭐ {seed["rating"]}</div>'
            f'<div style="margin-top:6px;">{badges_html(seed["genres"], limit=5)}</div></div>',
            unsafe_allow_html=True,
        )

    with right:
        recs = recommend(df, similarity_matrix, seed_title, top_n=top_n)
        if not recs:
            st.info("No close matches found in this catalog yet.")
        for r in recs:
            shared_bits = []
            if r["shared_genres"]:
                shared_bits.append("Shares " + ", ".join(r["shared_genres"]))
            if r["shared_tags"]:
                shared_bits.append("Themes: " + ", ".join(r["shared_tags"]))
            shared_line = " · ".join(shared_bits) if shared_bits else "Loosely related"

            rc1, rc2 = st.columns([3, 1])
            with rc1:
                st.markdown(f"**{r['title']}** ({r['year']}) · ⭐ {r['rating']}")
                st.caption(shared_line)
            with rc2:
                st.progress(r["match_score"] / 100, text=f"{r['match_score']}%")

    st.divider()
    st.markdown("**Match score breakdown**")
    if recs:
        fig = go.Figure(go.Bar(
            x=[r["match_score"] for r in recs][::-1],
            y=[r["title"] for r in recs][::-1],
            orientation="h",
            marker_color=GOLD,
        ))
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F3EEE4", xaxis=dict(range=[0, 100], gridcolor=RULE),
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Analytics tab -----------------------------------------------------------
with tab_analytics:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("**Titles per genre**")
        gc = genre_counts.sort_values()
        fig = go.Figure(go.Bar(x=gc.values, y=gc.index, orientation="h", marker_color=TEAL))
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F3EEE4", xaxis=dict(gridcolor=RULE),
        )
        st.plotly_chart(fig, use_container_width=True)

    with a2:
        st.markdown("**Rating distribution**")
        fig = px.histogram(df, x="rating", nbins=8, color_discrete_sequence=[ROSE])
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F3EEE4", xaxis=dict(gridcolor=RULE), yaxis=dict(gridcolor=RULE),
            bargap=0.1,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top 5 rated titles**")
    top5 = df.sort_values("rating", ascending=False).head(5).sort_values("rating")
    fig = go.Figure(go.Bar(x=top5["rating"], y=top5["title"], orientation="h", marker_color=GOLD))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F3EEE4", xaxis=dict(range=[7, 10], gridcolor=RULE),
    )
    st.plotly_chart(fig, use_container_width=True)
