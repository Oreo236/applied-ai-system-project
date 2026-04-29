"""
Agentic workflow module.

The LLM operates in a tool-use loop with three tools:
  - get_recommendations : runs the full content-based scorer
  - search_songs        : lightweight catalog filter for exploration
  - evaluate_diversity  : checks genre/mood spread of a candidate list

Workflow the model is instructed to follow:
  1. Call get_recommendations with inferred preferences.
  2. Call evaluate_diversity on the results.
  3. If diversity_score < 2.0, call search_songs to surface alternatives and
     call get_recommendations again with adjusted preferences.
  4. Produce a final natural-language answer.
"""
import json
from typing import Any, Dict, List

from groq import Groq

from recommender import recommend_songs

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI/Groq function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": (
                "Retrieve top-k songs from the catalog using structured user preferences. "
                "Returns a list of songs with scores and explanations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preferences": {
                        "type": "object",
                        "description": (
                            "Preference dict. Optional keys: genre (str), mood (str), "
                            "energy (float 0–1), likes_acoustic (bool), likes_popular (bool), "
                            "preferred_decade (int), detailed_mood (str), "
                            "likes_instrumental (bool), likes_live (bool)."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of songs to return (default 5).",
                    },
                },
                "required": ["preferences"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_songs",
            "description": (
                "Filter the catalog to explore what songs are available. "
                "Useful before or after get_recommendations to understand the data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {"type": "string", "description": "Filter by genre (case-insensitive)."},
                    "mood": {"type": "string", "description": "Filter by mood (case-insensitive)."},
                    "min_energy": {"type": "number", "description": "Minimum energy level 0–1."},
                    "max_energy": {"type": "number", "description": "Maximum energy level 0–1."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_diversity",
            "description": (
                "Measure the diversity of a set of song titles. "
                "Returns genre count, mood count, energy range, and a composite diversity_score. "
                "A score below 2.0 suggests the list is too homogeneous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "song_titles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of song titles to evaluate.",
                    }
                },
                "required": ["song_titles"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_tool(name: str, inp: Dict, songs: List[Dict]) -> Any:
    if name == "get_recommendations":
        prefs = inp.get("preferences", {})
        k = inp.get("k", 5)
        results = recommend_songs(prefs, songs, k=k)
        return [
            {
                "title": s["title"],
                "artist": s["artist"],
                "genre": s["genre"],
                "mood": s["mood"],
                "energy": s["energy"],
                "score": round(score, 2),
                "explanation": expl,
            }
            for s, score, expl in results
        ]

    if name == "search_songs":
        out = []
        for song in songs:
            if "genre" in inp and song["genre"].lower() != inp["genre"].lower():
                continue
            if "mood" in inp and song["mood"].lower() != inp["mood"].lower():
                continue
            if "min_energy" in inp and song["energy"] < inp["min_energy"]:
                continue
            if "max_energy" in inp and song["energy"] > inp["max_energy"]:
                continue
            out.append(
                {
                    "title": song["title"],
                    "artist": song["artist"],
                    "genre": song["genre"],
                    "mood": song["mood"],
                    "energy": song["energy"],
                }
            )
        return out

    if name == "evaluate_diversity":
        titles = set(inp.get("song_titles", []))
        matched = [s for s in songs if s["title"] in titles]
        if not matched:
            return {"error": "No matching songs found in catalog"}
        genres = list({s["genre"] for s in matched})
        moods = list({s["mood"] for s in matched})
        energies = [s["energy"] for s in matched]
        energy_range = round(max(energies) - min(energies), 2)
        diversity_score = round(
            (len(genres) * 0.4 + len(moods) * 0.4 + energy_range * 2.0), 2
        )
        return {
            "unique_genres": genres,
            "genre_count": len(genres),
            "unique_moods": moods,
            "mood_count": len(moods),
            "energy_min": round(min(energies), 2),
            "energy_max": round(max(energies), 2),
            "energy_range": energy_range,
            "diversity_score": diversity_score,
            "verdict": "diverse" if diversity_score >= 2.0 else "homogeneous — consider broadening",
        }

    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a music recommendation agent. You have tools available to find and evaluate songs.

To answer the user, you must use the tools in this order:
- First use get_recommendations to retrieve songs matching the user's preferences.
- Then use evaluate_diversity on those song titles.
- If diversity_score is below 2.0, use search_songs to explore alternatives, then use get_recommendations again with broader preferences.
- Once you have good recommendations, write a warm paragraph summarising your top picks.

Use the tools provided. Do not write your final answer until you have used the tools."""


def run_agent(
    query: str, songs: List[Dict], api_key: str, max_iters: int = 8
) -> Dict:
    """
    Run the agentic recommendation loop.

    Returns a dict with:
      query            — original user query
      final_text       — model's narrative response
      tool_trace       — list of {tool, input, output} dicts (full audit trail)
      recommendations  — final list of recommended songs
    """
    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": query},
    ]
    tool_trace: List[Dict] = []
    final_text = ""

    for _ in range(max_iters):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Add assistant turn to history
        msg_dict: Dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if finish_reason == "stop" or not msg.tool_calls:
            final_text = msg.content or ""
            break

        # Execute tools and feed results back
        for tc in msg.tool_calls:
            inp = json.loads(tc.function.arguments)
            result = _execute_tool(tc.function.name, inp, songs)
            tool_trace.append({"tool": tc.function.name, "input": inp, "output": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    final_recs: List[Dict] = []
    for entry in reversed(tool_trace):
        if entry["tool"] == "get_recommendations" and isinstance(entry["output"], list):
            final_recs = entry["output"]
            break

    return {
        "query": query,
        "final_text": final_text,
        "tool_trace": tool_trace,
        "recommendations": final_recs,
    }
