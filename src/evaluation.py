"""
Reliability and evaluation module.

Provides three capabilities:
  measure_diversity        — genre/mood spread and energy range for a result set
  run_consistency_check    — verifies the recommender is deterministic
  run_benchmark            — runs a suite of profiles and collects aggregate metrics
"""
from typing import Dict, List, Tuple

from recommender import recommend_songs


def measure_diversity(recommendations: List[Tuple]) -> Dict:
    """
    Compute diversity metrics for a list of (song_dict, score, explanation) tuples.
    """
    if not recommendations:
        return {"error": "empty recommendation list"}
    songs = [r[0] for r in recommendations]
    scores = [r[1] for r in recommendations]
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})
    energies = [s["energy"] for s in songs]
    energy_range = round(max(energies) - min(energies), 3)
    # Composite: each unique genre/mood counts for 0.4, energy range for up to 2.0
    diversity_score = round(len(genres) * 0.4 + len(moods) * 0.4 + energy_range * 2.0, 2)
    return {
        "genre_spread": genres,
        "genre_count": len(genres),
        "mood_spread": moods,
        "mood_count": len(moods),
        "energy_range": [round(min(energies), 3), round(max(energies), 3)],
        "score_range": [round(min(scores), 2), round(max(scores), 2)],
        "diversity_score": diversity_score,
        "top_song": songs[0]["title"],
    }


def run_consistency_check(
    preferences: Dict, songs: List[Dict], n_runs: int = 3
) -> Dict:
    """
    Run the same profile n_runs times and verify results are identical.

    The content-based scorer is deterministic, so this should always pass.
    If it fails it indicates a bug (e.g. non-deterministic sort key).
    """
    results = []
    for _ in range(n_runs):
        recs = recommend_songs(preferences, songs, k=5)
        results.append([r[0]["title"] for r in recs])

    all_same = all(r == results[0] for r in results)
    return {
        "consistent": all_same,
        "n_runs": n_runs,
        "all_results": results,
        "verdict": (
            "PASS — deterministic: same input always produces identical output."
            if all_same
            else "FAIL — non-deterministic results detected!"
        ),
    }


def run_benchmark(profiles: List[Tuple], songs: List[Dict]) -> Dict:
    """
    Run every profile in the suite, collect per-profile metrics, and compute
    aggregate statistics: average top score, average genre spread, and catalog
    coverage (what fraction of the catalog ever appears in recommendations).
    """
    profile_reports = []
    top_scores: List[float] = []
    genre_spreads: List[int] = []
    songs_ever_recommended: set = set()

    for prefs, label in profiles:
        recs = recommend_songs(prefs, songs, k=5)
        diversity = measure_diversity(recs)
        top_score = recs[0][1] if recs else 0.0
        top_scores.append(top_score)
        genre_spreads.append(diversity.get("genre_count", 0))
        for r in recs:
            songs_ever_recommended.add(r[0]["title"])
        consistency = run_consistency_check(prefs, songs, n_runs=2)
        profile_reports.append(
            {
                "label": label,
                "top_score": round(top_score, 2),
                "diversity": diversity,
                "consistent": consistency["consistent"],
            }
        )

    n = len(profiles)
    coverage_count = len(songs_ever_recommended)
    return {
        "total_profiles": n,
        "profiles": profile_reports,
        "aggregate": {
            "avg_top_score": round(sum(top_scores) / n, 2) if n else 0.0,
            "avg_genre_spread": round(sum(genre_spreads) / n, 2) if n else 0.0,
            "songs_recommended_count": coverage_count,
            "catalog_size": len(songs),
            "catalog_coverage_pct": (
                round(coverage_count / len(songs) * 100, 1) if songs else 0.0
            ),
            "songs_never_recommended": sorted(
                {s["title"] for s in songs} - songs_ever_recommended
            ),
        },
    }
