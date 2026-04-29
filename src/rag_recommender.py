"""
RAG (Retrieval-Augmented Generation) module.

Pipeline:
    1. Groq parses a natural-language query into structured preferences.
    2. The existing content-based scorer retrieves the top-k matching songs.
    3. Those retrieved songs become the context the model uses to write the final response.

The AI never invents songs — it only explains what the scorer retrieved.
"""
import json
from typing import Dict, List, Tuple

from groq import Groq

from recommender import recommend_songs


# ---------------------------------------------------------------------------
# Step 1 — preference extraction
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """\
You are a music preference parser. Extract structured preferences from the user's request.
Return ONLY a valid JSON object. Use only these keys:
  genre         (string) — e.g. "pop", "rock", "lofi", "jazz", "edm", "ambient",
                           "hip-hop", "r&b", "classical", "synthwave", "indie pop",
                           "country", "folk", "metal", "soul"
  mood          (string) — e.g. "happy", "chill", "intense", "relaxed", "focused",
                           "moody", "euphoric", "peaceful", "romantic", "energetic",
                           "nostalgic", "melancholic", "uplifting", "aggressive"
  energy        (float 0.0–1.0)
  likes_acoustic (bool)
  likes_popular  (bool)  — true = mainstream, false = underground
  preferred_decade (int) — e.g. 1980, 1990, 2000, 2010, 2020
  detailed_mood  (string) — same values as mood
  likes_instrumental (bool) — true = no vocals
  likes_live     (bool)

Only include keys you can confidently infer. Return only valid JSON, no commentary."""


def parse_preferences(query: str, api_key: str) -> Dict:
    """Use the LLM to extract a structured preference dict from free text."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=256,
        messages=[
            {"role": "system", "content": _PARSE_SYSTEM},
            {"role": "user", "content": query},
        ],
    )
    try:
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError, AttributeError):
        return {}


# ---------------------------------------------------------------------------
# Step 3 — response generation
# ---------------------------------------------------------------------------

_RESPOND_SYSTEM = """\
You are a friendly music recommendation assistant. Songs from a catalog have already been
retrieved and scored for the user. Describe those recommendations warmly and concisely:
2–3 sentences per song explaining why it fits what they asked for.
Do not invent songs or features — only reference what is listed."""


def _build_context(recommendations: List[Tuple]) -> str:
    lines = []
    for rank, (song, score, explanation) in enumerate(recommendations, 1):
        lines.append(
            f"{rank}. \"{song['title']}\" by {song['artist']} "
            f"(genre: {song['genre']}, mood: {song['mood']}, "
            f"energy: {song['energy']:.2f}, score: {score:.2f})\n"
            f"   Why it ranked here: {explanation}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rag_recommend(
    query: str,
    songs: List[Dict],
    api_key: str,
    k: int = 5,
) -> Tuple[List, Dict, str]:
    """
    Full RAG pipeline. Returns (recommendations, parsed_preferences, natural_language_response).

    The retrieval step uses the deterministic content-based scorer so results are
    reproducible; the LLM only generates the explanatory text.
    """
    preferences = parse_preferences(query, api_key)
    recommendations = recommend_songs(preferences, songs, k=k)

    context = _build_context(recommendations)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=600,
        messages=[
            {"role": "system", "content": _RESPOND_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"User request: {query}\n\n"
                    f"Retrieved songs from catalog:\n{context}"
                ),
            },
        ],
    )
    natural_response = response.choices[0].message.content
    return recommendations, preferences, natural_response
