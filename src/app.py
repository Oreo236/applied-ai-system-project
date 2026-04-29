"""
AI Music Recommender — Streamlit application.

Three tabs:
  1. RAG Chat        — natural-language chat backed by retrieval-augmented generation
  2. Agent Mode      — agentic workflow with tool-use trace
  3. Reliability     — benchmark, consistency checks, and session coverage stats
"""
import os
import sys
from pathlib import Path

# Ensure src/ is on the path regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
from key_manager import get_api_key

from recommender import load_songs
from rag_recommender import rag_recommend, parse_preferences
from agent import run_agent
from evaluation import measure_diversity, run_consistency_check, run_benchmark
from logger import RecommendationLogger

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Music Recommender",
    page_icon="🎵",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — API key and data loading
# ---------------------------------------------------------------------------

st.sidebar.title("AI Music Recommender")
st.sidebar.markdown("Powered by Groq + content-based retrieval")

# Load environment variables from a local .env (local dev convenience)
load_dotenv()
api_key = get_api_key()

DATA_PATH = Path(__file__).parent.parent / "data" / "songs.csv"


@st.cache_data
def load_catalog():
    return load_songs(str(DATA_PATH))


songs = load_catalog()
logger = RecommendationLogger()

st.sidebar.markdown(f"**Catalog:** {len(songs)} songs loaded")

DIAGRAM_PATH = Path(__file__).parent.parent / "assets" / "flowchart.png"
with st.sidebar.expander("How it works", expanded=False):
    if DIAGRAM_PATH.exists():
        st.image(str(DIAGRAM_PATH))
    else:
        st.markdown(
            "**RAG Chat:** query → parse preferences → score catalog → LLM response\n\n"
            "**Agent Mode:** LLM tool-use loop → get_recommendations → evaluate_diversity "
            "→ (search_songs if needed) → final answer"
        )

if not api_key:
        st.warning(
            "AI features require GROQ_API_KEY set as an environment variable."
        )
        with st.expander("How to set the key locally"):
            st.markdown(
                "Create a file named `.env` with a line `GROQ_API_KEY=your_key_here`,\n"
                "or set the environment variable in your shell before starting Streamlit."
            )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_rag, tab_agent, tab_eval = st.tabs(["RAG Chat", "Agent Mode", "Reliability"])

# ───────────────────────────────────────────────
# TAB 1 — RAG Chat
# ───────────────────────────────────────────────
with tab_rag:
    st.header("RAG Chat")
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1 — Parse**\nLLM reads your query and extracts genre, mood, energy, and other preferences.")
    col2.info("**Step 2 — Retrieve**\nA deterministic scorer ranks every song in the catalog against those preferences.")
    col3.info("**Step 3 — Respond**\nLLM describes the top matches in plain English using only the retrieved songs as context.")

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    # Render history
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("retrieved"):
                with st.expander("Retrieved songs (RAG context)", expanded=False):
                    for song, score, expl in msg["retrieved"]:
                        st.markdown(
                            f"**{song['title']}** by {song['artist']}  "
                            f"*(score: {score:.2f})*"
                        )
                        st.caption(f"Why: {expl}")
            if msg.get("preferences"):
                with st.expander("Parsed preferences", expanded=False):
                    st.json(msg["preferences"])

    # Input
    user_input = st.chat_input(
        "What kind of music are you in the mood for?", key="rag_input"
    )
    if user_input:
        if not api_key:
            st.error("Please enter your API key first.")
        else:
            st.session_state.rag_messages.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving songs and generating response..."):
                    try:
                        recs, prefs, response_text = rag_recommend(
                            user_input, songs, api_key
                        )
                        st.markdown(response_text)
                        with st.expander("Retrieved songs (RAG context)", expanded=True):
                            for song, score, expl in recs:
                                st.markdown(
                                    f"**{song['title']}** by {song['artist']}  "
                                    f"*(score: {score:.2f})*"
                                )
                                st.caption(f"Why: {expl}")
                        with st.expander("Parsed preferences", expanded=False):
                            st.json(prefs)

                        logger.log_session(
                            user_input, prefs, recs, {"mode": "rag"}
                        )
                        st.session_state.rag_messages.append(
                            {
                                "role": "assistant",
                                "content": response_text,
                                "retrieved": recs,
                                "preferences": prefs,
                            }
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")

# ───────────────────────────────────────────────
# TAB 2 — Agent Mode
# ───────────────────────────────────────────────
with tab_agent:
    st.header("Agent Mode")
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**Step 1**\nCall `get_recommendations` with inferred preferences.")
    col2.info("**Step 2**\nCall `evaluate_diversity` — score ≥ 2.0 means the list is varied enough.")
    col3.info("**Step 3**\nIf too homogeneous, call `search_songs` and re-run recommendations with broader preferences.")
    col4.info("**Step 4**\nWrite a final narrative answer. Every tool call appears in the trace below.")

    agent_query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. something energetic for a workout but not too mainstream",
        key="agent_query",
    )
    run_btn = st.button("Run Agent", disabled=not api_key, key="agent_run")

    if run_btn and agent_query:
        with st.spinner("Agent working..."):
            try:
                result = run_agent(agent_query, songs, api_key)

                st.subheader("Agent Response")
                st.markdown(result["final_text"])

                st.subheader("Final Recommendations")
                if result["recommendations"]:
                    for i, rec in enumerate(result["recommendations"], 1):
                        st.markdown(
                            f"**{i}. {rec['title']}** by {rec['artist']}  "
                            f"*(score: {rec['score']:.2f})*"
                        )
                        st.caption(
                            f"Genre: {rec['genre']} | Mood: {rec['mood']} | "
                            f"Energy: {rec['energy']:.2f}"
                        )
                else:
                    st.info("No recommendations returned.")

                st.subheader("Tool Trace")
                for step_num, step in enumerate(result["tool_trace"], 1):
                    with st.expander(
                        f"Step {step_num}: `{step['tool']}`", expanded=False
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Input**")
                            st.json(step["input"])
                        with col2:
                            st.markdown("**Output**")
                            st.json(step["output"])

                # Log with agent metadata
                recs_for_log = [
                    (
                        next(
                            (s for s in songs if s["title"] == r["title"]),
                            {"title": r["title"], "artist": r["artist"],
                             "genre": "", "score": r["score"]},
                        ),
                        r["score"],
                        r.get("explanation", ""),
                    )
                    for r in result["recommendations"]
                ]
                logger.log_session(
                    agent_query,
                    {},
                    recs_for_log,
                    {
                        "mode": "agent",
                        "tool_calls": len(result["tool_trace"]),
                    },
                )
            except Exception as e:
                st.error(f"Error: {e}")

# ───────────────────────────────────────────────
# TAB 3 — Reliability
# ───────────────────────────────────────────────
with tab_eval:
    st.header("Reliability & Evaluation")
    col1, col2, col3 = st.columns(3)
    col1.info("**Consistency Check**\nRuns the same profile 3 times and verifies the results are identical. No API key needed.")
    col2.info("**Full Benchmark**\nRuns 16 profiles (4 standard + 12 adversarial) and reports average score, genre spread, and catalog coverage.")
    col3.info("**Session Coverage**\nTracks how much of the catalog has been surfaced across all logged sessions.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Consistency Check")
        st.markdown("Verify that the scorer always returns the same result for the same input.")
        profile_options = {
            "Pop / happy / high energy": {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False},
            "Lofi / chill / acoustic":   {"genre": "lofi", "mood": "chill", "energy": 0.38, "likes_acoustic": True},
            "Rock / intense":             {"genre": "rock", "mood": "intense", "energy": 0.91, "likes_acoustic": False},
            "Empty profile":              {},
        }
        selected_label = st.selectbox("Select a profile", list(profile_options.keys()))
        if st.button("Check Consistency"):
            prefs = profile_options[selected_label]
            result = run_consistency_check(prefs, songs)
            if result["consistent"]:
                st.success(result["verdict"])
            else:
                st.error(result["verdict"])
            with st.expander("All run results"):
                for i, run in enumerate(result["all_results"], 1):
                    st.write(f"Run {i}: {run}")

    with col_b:
        st.subheader("Session Coverage")
        st.markdown("How much of the catalog has been surfaced across all logged sessions?")
        stats = logger.get_coverage_stats(len(songs))
        st.metric("Total Sessions Logged", stats["total_sessions"])
        st.metric("Catalog Coverage", f"{stats['coverage_pct']}%")
        st.metric("Unique Songs Recommended", stats["unique_songs_recommended"])
        st.metric("Songs Never Recommended", stats["songs_never_recommended"])

    st.divider()
    st.subheader("Full Benchmark Suite")
    st.markdown(
        "Runs all 16 built-in profiles and measures top score, genre diversity, "
        "and catalog coverage."
    )

    # Import profiles from main without executing main()
    benchmark_profiles = [
        ({"genre": "pop",  "mood": "happy",   "energy": 0.80, "likes_acoustic": False}, "Baseline — pop/happy/0.8"),
        ({"genre": "pop",  "mood": "happy",   "energy": 0.85, "likes_acoustic": False}, "Baseline 2 — pop/happy/0.85"),
        ({"genre": "lofi", "mood": "chill",   "energy": 0.38, "likes_acoustic": True},  "Baseline 3 — lofi/chill/acoustic"),
        ({"genre": "rock", "mood": "intense", "energy": 0.92, "likes_acoustic": False}, "Baseline 4 — rock/intense"),
        ({"genre": "pop",  "mood": "sad",     "energy": 0.80, "likes_acoustic": False}, "ADV 1 — unknown mood 'sad'"),
        ({"genre": "lofi", "mood": "chill",   "energy": 0.90, "likes_acoustic": True},  "ADV 2 — high energy + acoustic lofi"),
        ({"genre": "pop",  "mood": "happy",   "energy": 1.50, "likes_acoustic": False}, "ADV 3 — out-of-range energy 1.5"),
        ({"genre": "Pop",  "mood": "Happy",   "energy": 0.80, "likes_acoustic": False}, "ADV 4 — wrong capitalisation"),
        ({"genre": "metal","mood": "angry",   "energy": 0.97, "likes_acoustic": False}, "ADV 5 — absent genre/mood"),
        ({},                                                                              "ADV 6 — empty profile"),
        ({"genre": "edm", "mood": "euphoric", "energy": 0.96, "likes_acoustic": False,
          "likes_popular": True, "preferred_decade": 2010,
          "detailed_mood": "euphoric", "likes_instrumental": False, "likes_live": False}, "NEW 1 — full EDM profile"),
        ({"genre": "lofi","mood": "chill",    "energy": 0.38, "likes_acoustic": True,
          "likes_popular": False, "preferred_decade": 2020,
          "detailed_mood": "focused", "likes_instrumental": True, "likes_live": False},  "NEW 2 — underground lofi/instrumental"),
        ({"genre": "rock","mood": "intense",  "energy": 0.91, "likes_acoustic": False,
          "likes_popular": True, "preferred_decade": 1980,
          "detailed_mood": "aggressive","likes_instrumental": False, "likes_live": True}, "NEW 3 — nostalgic 1980s rock"),
        ({"genre": "jazz","mood": "relaxed",  "energy": 0.37, "likes_acoustic": True,
          "likes_popular": True, "preferred_decade": 1990,
          "detailed_mood": "romantic", "likes_instrumental": False, "likes_live": True},  "NEW 4 — jazz cafe"),
        ({"preferred_decade": 2020},                                                      "NEW 5 — decade only: 2020s"),
        ({"likes_popular": True, "preferred_decade": 1960},                               "NEW 6 — popular + 1960s"),
    ]

    if st.button("Run Full Benchmark"):
        with st.spinner("Running 16 profiles..."):
            report = run_benchmark(benchmark_profiles, songs)

        agg = report["aggregate"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Profiles Run", report["total_profiles"])
        c2.metric("Avg Top Score", f"{agg['avg_top_score']:.2f}")
        c3.metric("Avg Genre Spread", f"{agg['avg_genre_spread']:.1f}")
        c4.metric("Catalog Coverage", f"{agg['catalog_coverage_pct']}%")

        st.markdown(
            f"**Songs never recommended:** "
            + (", ".join(agg["songs_never_recommended"]) or "All songs were recommended at least once!")
        )

        with st.expander("Per-profile breakdown", expanded=False):
            for p in report["profiles"]:
                consistency_icon = "✅" if p["consistent"] else "❌"
                st.markdown(
                    f"{consistency_icon} **{p['label']}** — "
                    f"top score: {p['top_score']:.2f}, "
                    f"genres: {p['diversity']['genre_count']}, "
                    f"moods: {p['diversity']['mood_count']}"
                )
