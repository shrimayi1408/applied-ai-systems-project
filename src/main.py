"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import argparse
import logging

from src.recommender import load_songs, recommend_songs, MAX_SCORE
from src.dj_model import DJNova


PROFILES = {
    "Starter (pop/happy)": {"genre": "pop", "mood": "happy", "energy": 0.8},
    # Adversarial / edge-case profiles
    "Edge 1: high energy + sad mood": {"energy": 0.9, "mood": "sad"},
    "Edge 3: unknown genre/mood": {"genre": "k-pop", "mood": "ecstatic"},
    "Edge 10: likes_acoustic vs acousticness 0.0": {
        "likes_acoustic": True,
        "acousticness": 0.0,
    },
}


def print_recommendations(name: str, user_prefs: dict, songs: list, dj=None, k: int = 5) -> None:
    recommendations = recommend_songs(user_prefs, songs, k=k)

    width = 60
    print("\n" + "=" * width)
    print(f"{name:^{width}}")
    print(
        f"{'Profile: ' + ', '.join(f'{key}={value}' for key, value in user_prefs.items()):^{width}}"
    )
    print("=" * width)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n#{rank}  {song['title']}  -  {song['artist']}")
        print(f"    Score: {score:.2f} / {MAX_SCORE:g}")
        if dj is not None:
            print(f"    DJ Nova: {dj.narrate(user_prefs, song, rank)}")
        print("    Why:")
        for reason in explanation.split("; "):
            print(f"      - {reason}")
        print("-" * width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Music recommender with a fine-tuned DJ voice")
    parser.add_argument("--plain", action="store_true", help="skip the DJ Nova model, show scores only")
    args = parser.parse_args()
    logging.basicConfig(filename="dj_nova.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    songs = load_songs("data/songs.csv")
    dj = None if args.plain else DJNova()
    for name, prefs in PROFILES.items():
        print_recommendations(name, prefs, songs, dj)


if __name__ == "__main__":
    main()
