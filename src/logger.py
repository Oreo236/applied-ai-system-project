import json
import os
from datetime import datetime
from typing import Dict, List

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


class RecommendationLogger:
    """Persists every recommendation session to a JSONL file for auditing and analysis."""

    def __init__(self, log_file: str = "sessions.jsonl"):
        os.makedirs(LOGS_DIR, exist_ok=True)
        self.log_path = os.path.join(LOGS_DIR, log_file)

    def log_session(
        self,
        query: str,
        preferences: Dict,
        recommendations: List,
        metadata: Dict = None,
    ) -> Dict:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "preferences": preferences,
            "recommendations": [
                {
                    "title": r[0]["title"],
                    "artist": r[0]["artist"],
                    "genre": r[0]["genre"],
                    "score": round(r[1], 3),
                }
                for r in recommendations
            ],
            "metadata": metadata or {},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def load_sessions(self) -> List[Dict]:
        if not os.path.exists(self.log_path):
            return []
        sessions = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sessions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return sessions

    def get_coverage_stats(self, total_songs: int) -> Dict:
        sessions = self.load_sessions()
        recommended_titles = set()
        for s in sessions:
            for r in s.get("recommendations", []):
                recommended_titles.add(r["title"])
        return {
            "total_sessions": len(sessions),
            "unique_songs_recommended": len(recommended_titles),
            "catalog_size": total_songs,
            "coverage_pct": (
                round(len(recommended_titles) / total_songs * 100, 1)
                if total_songs
                else 0.0
            ),
            "songs_never_recommended": total_songs - len(recommended_titles),
        }
