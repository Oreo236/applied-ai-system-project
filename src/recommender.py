import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: int = 50
    release_decade: int = 2000
    detailed_mood: str = ""
    instrumentalness: float = 0.0
    liveness: float = 0.0

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Parse a CSV file of songs and return a list of dicts with typed numeric fields."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":               int(row["id"]),
                "title":            row["title"],
                "artist":           row["artist"],
                "genre":            row["genre"],
                "mood":             row["mood"],
                "energy":           float(row["energy"]),
                "tempo_bpm":        float(row["tempo_bpm"]),
                "valence":          float(row["valence"]),
                "danceability":     float(row["danceability"]),
                "acousticness":     float(row["acousticness"]),
                "popularity":       int(row["popularity"]),
                "release_decade":   int(row["release_decade"]),
                "detailed_mood":    row["detailed_mood"],
                "instrumentalness": float(row["instrumentalness"]),
                "liveness":         float(row["liveness"]),
            })
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences and return (total_score, reason_strings)."""
    score = 0.0
    reasons = []

    # Genre match — exact, weight 1.5 (halved from 3.0 to reduce categorical dominance)
    if song["genre"].lower() == (user_prefs.get("genre") or "").lower():
        score += 1.5
        reasons.append(f"matched genre: {song['genre']}")

    # Mood match — exact, weight 2.5
    if song["mood"].lower() == (user_prefs.get("mood") or "").lower():
        score += 2.5
        reasons.append(f"matched mood: {song['mood']}")

    # Energy proximity — squared distance, weight 4.0 (doubled from 2.0 to amplify continuous mismatch)
    if user_prefs.get("energy") is not None:
        energy_sim = 1 - (song["energy"] - user_prefs["energy"]) ** 2
        score += energy_sim * 4.0
        reasons.append(f"energy {song['energy']:.2f} vs your target {user_prefs['energy']:.2f}")

    # Acousticness proximity — squared distance, weight 1.5
    # likes_acoustic=True targets 0.80, False targets 0.15
    if user_prefs.get("likes_acoustic") is not None:
        target_acoustic = 0.80 if user_prefs["likes_acoustic"] else 0.15
        acoustic_sim = 1 - (song["acousticness"] - target_acoustic) ** 2
        score += acoustic_sim * 1.5
        label = "acoustic" if user_prefs["likes_acoustic"] else "non-acoustic"
        reasons.append(f"acousticness {song['acousticness']:.2f} suits a {label} preference")

    # Valence proximity — squared distance against fixed 0.70, weight 0.5
    valence_sim = 1 - (song["valence"] - 0.70) ** 2
    score += valence_sim * 0.5
    reasons.append(f"valence {song['valence']:.2f}")

    # Popularity affinity — user preference key: "likes_popular" (bool), weight 1.5
    if user_prefs.get("likes_popular") is not None:
        likes_popular = user_prefs["likes_popular"]
        if likes_popular:
            score += (song["popularity"] / 100) * 1.5
        else:
            score += ((100 - song["popularity"]) / 100) * 1.5
        reasons.append(
            f"popularity {song['popularity']}/100 suits a "
            f"{'popular' if likes_popular else 'underground'} preference"
        )

    # Decade match — user preference key: "preferred_decade" (int), weight 1.0 / 0.5
    if user_prefs.get("preferred_decade") is not None:
        preferred_decade = user_prefs["preferred_decade"]
        decade_diff = abs(song["release_decade"] - preferred_decade)
        if decade_diff == 0:
            score += 1.0
        elif decade_diff == 10:
            score += 0.5
        reasons.append(
            f"release decade {song['release_decade']} vs your preferred {preferred_decade}"
        )

    # Detailed mood match — user preference key: "detailed_mood" (str), weight 2.0
    if user_prefs.get("detailed_mood") is not None:
        if song["detailed_mood"].lower() == user_prefs["detailed_mood"].lower():
            score += 2.0
            reasons.append(f"matched detailed mood: {song['detailed_mood']}")

    # Instrumentalness proximity — user preference key: "likes_instrumental" (bool), weight 1.0
    if user_prefs.get("likes_instrumental") is not None:
        likes_instrumental = user_prefs["likes_instrumental"]
        target = 0.85 if likes_instrumental else 0.10
        score += (1 - (song["instrumentalness"] - target) ** 2) * 1.0
        reasons.append(
            f"instrumentalness {song['instrumentalness']:.2f} suits a "
            f"{'instrumental' if likes_instrumental else 'vocal'} preference"
        )

    # Liveness proximity — user preference key: "likes_live" (bool), weight 0.5
    if user_prefs.get("likes_live") is not None:
        likes_live = user_prefs["likes_live"]
        target = 0.75 if likes_live else 0.05
        score += (1 - (song["liveness"] - target) ** 2) * 0.5
        reasons.append(
            f"liveness {song['liveness']:.2f} suits a "
            f"{'live' if likes_live else 'studio'} preference"
        )

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song, sort by score descending, and return the top k as (song, score, explanation)."""
    scored = [
        (song, score, ", ".join(reasons))
        for song in songs
        for score, reasons in [score_song(user_prefs, song)]
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
