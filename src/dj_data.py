"""Training-data builder for the 'DJ Nova' voice model.

Maps structured recommendation facts -> a blurb in a consistent persona voice
(warm late-night radio host: short punchy sentences, second person, one sign-off).
The fine-tuned model learns the *voice*; the facts always come from the scorer.
"""
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

PROMPT_PREFIX = "dj nova: "


def energy_word(e: float) -> str:
    if e < 0.35:
        return "barely-a-whisper"
    if e < 0.55:
        return "laid-back"
    if e < 0.75:
        return "steady"
    if e < 0.88:
        return "high-voltage"
    return "full-throttle"


def build_prompt(song: Dict, mood_match: bool, genre_match: bool, energy_fit: str, rank: int) -> str:
    """Serialize facts into the model's input string (used for training AND inference)."""
    return (
        f"{PROMPT_PREFIX}genre={song['genre']} | "
        f"mood={song['mood']} | feel={energy_word(float(song['energy']))} | "
        f"mood_match={'yes' if mood_match else 'no'} | genre_match={'yes' if genre_match else 'no'} | "
        f"energy_fit={energy_fit} | rank={rank}"
    )


OPENERS = {
    1: ["Your number one tonight:", "Top of the stack, friend:", "Kicking things off right:"],
    2: ["Next up:", "Stay with me, here comes", "Keep the dial right here for"],
    3: ["Spinning in the middle slot:", "Don't touch that knob, it's", "Coming in hot:"],
}
FALLBACK_OPENERS = ["Also on the turntable:", "One more for the road:", "Dropping the needle on"]
MOOD_LINES = [
    "It hits that {mood} mood you asked for.",
    "That {mood} vibe? Right on the money.",
    "Pure {mood}, exactly your speed.",
]
NO_MOOD_LINES = [
    "It's not quite the {mood} you ordered, but hear me out.",
    "A little off your mood, but it earns its spot.",
]
GENRE_LINES = ["{genre} through and through.", "Proper {genre}, just how you like it."]
ENERGY_LINES = {
    "tight": ["The energy sits right where you wanted it.", "Energy-wise? Bullseye."],
    "close": ["The energy is in the neighborhood.", "Energy's close enough to feel right."],
    "far": ["Fair warning, the energy wanders from your target.", "The energy strays a bit, but go with it."],
}
SIGNOFFS = ["Stay golden.", "This is DJ Nova, keeping the night warm.", "Turn it up, you've earned it.", "Trust me on this one."]


def write_blurb(song: Dict, mood_match: bool, genre_match: bool, energy_fit: str, rank: int, rng: random.Random) -> str:
    opener = rng.choice(OPENERS.get(rank, FALLBACK_OPENERS))
    parts = [f"{opener} [TITLE] by [ARTIST].", f"A {energy_word(float(song['energy']))} {song['genre']} track."]
    parts.append(rng.choice(MOOD_LINES if mood_match else NO_MOOD_LINES).format(mood=song["mood"]))
    if genre_match:
        parts.append(rng.choice(GENRE_LINES).format(genre=song["genre"]))
    parts.append(rng.choice(ENERGY_LINES[energy_fit]))
    parts.append(rng.choice(SIGNOFFS))
    return " ".join(parts)


def fill_names(text: str, song: Dict) -> str:
    """Names are copied by code, never by the model (small models garble unseen names)."""
    return text.replace("[TITLE]", song["title"]).replace("[ARTIST]", song["artist"])


def load_rows(csv_path: str) -> List[Dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_examples(songs: List[Dict], per_song: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    out = []
    for song in songs:
        for _ in range(per_song):
            mm, gm = rng.random() < 0.5, rng.random() < 0.5
            fit = rng.choice(["tight", "close", "far"])
            rank = rng.randint(1, 5)
            out.append({
                "title": song["title"],  # metadata only, not model input
                "input": build_prompt(song, mm, gm, fit, rank),
                "target": write_blurb(song, mm, gm, fit, rank, rng),
            })
    return out


def main(csv_path: str = "data/songs.csv", out_dir: str = "data/dj") -> None:
    rows = load_rows(csv_path)
    held_out = rows[-4:]  # unseen songs: tests whether the voice generalizes
    train_rows = rows[:-4]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, data in (("train", build_examples(train_rows, 40, 0)), ("heldout", build_examples(held_out, 10, 1))):
        with open(f"{out_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"wrote {len(data)} examples -> {out_dir}/{name}.jsonl")


if __name__ == "__main__":
    main()
