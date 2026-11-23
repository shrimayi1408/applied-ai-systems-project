"""DJ Nova: runtime wrapper around the fine-tuned voice model, with logging and guardrails."""
import logging
import os
from typing import Dict, Optional

from src.dj_data import build_prompt, fill_names

log = logging.getLogger("dj_nova")
MODEL_DIR = os.environ.get("DJ_MODEL_DIR", "models/dj_nova")
MAX_CHARS = 600


def derive_facts(user_prefs: Dict, song: Dict):
    """Ground truth about WHY a song was picked, computed from the scorer's inputs (not the model)."""
    mood_match = str(song["mood"]).lower() == str(user_prefs.get("mood", "")).lower()
    genre_match = str(song["genre"]).lower() == str(user_prefs.get("genre", "")).lower()
    target = user_prefs.get("energy")
    gap = abs(float(song["energy"]) - target) if target is not None else 0.15
    fit = "tight" if gap < 0.1 else "close" if gap < 0.25 else "far"
    return mood_match, genre_match, fit


def plain_blurb(song: Dict, mood_match: bool, genre_match: bool, fit: str) -> str:
    """Deterministic fallback used whenever the model is missing or its output fails validation."""
    bits = [f"{song['title']} by {song['artist']} ({song['genre']}, {song['mood']})."]
    bits.append("Mood matches." if mood_match else "Mood differs from your request.")
    if genre_match:
        bits.append("Genre matches.")
    bits.append(f"Energy fit: {fit}.")
    return " ".join(bits)


def validate(text: str, song: Dict, mood_match: bool, genre_match: bool) -> Optional[str]:
    """Return a failure reason, or None if the generated blurb is faithful and safe to show."""
    if not text or len(text) > MAX_CHARS:
        return "empty or too long"
    low = text.lower()
    if song["title"].lower() not in low or song["artist"].lower() not in low:
        return "missing title/artist (possible hallucination)"
    if not mood_match and f"that {song['mood']} vibe".lower() in low:
        return "claims a mood match that did not happen"
    if mood_match and song["mood"].lower() not in low:
        return "omits the matched mood"
    return None


class DJNova:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model = self.tok = None
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).eval()
            log.info("loaded fine-tuned model from %s", model_dir)
        except Exception as exc:  # missing weights, no torch, corrupt files...
            log.warning("fine-tuned model unavailable (%s); using plain fallback text. "
                        "Run `python -m src.train_dj` to create it.", exc)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def generate(self, song: Dict, mood_match: bool, genre_match: bool, fit: str, rank: int) -> str:
        if self.ready:
            try:
                import torch
                prompt = build_prompt(song, mood_match, genre_match, fit, rank)
                enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    ids = self.model.generate(**enc, max_new_tokens=120, num_beams=4)
                text = fill_names(self.tok.decode(ids[0], skip_special_tokens=True).strip(), song)
                problem = validate(text, song, mood_match, genre_match)
                if problem is None:
                    log.info("rank=%d '%s': model blurb accepted", rank, song["title"])
                    return text
                log.warning("rank=%d '%s': model blurb rejected (%s); using fallback", rank, song["title"], problem)
            except Exception:
                log.exception("generation failed for '%s'; using fallback", song["title"])
        return plain_blurb(song, mood_match, genre_match, fit)

    def narrate(self, user_prefs: Dict, song: Dict, rank: int) -> str:
        mm, gm, fit = derive_facts(user_prefs, song)
        return self.generate(song, mm, gm, fit, rank)
