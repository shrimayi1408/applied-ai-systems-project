"""Compare base flan-t5-small vs the fine-tuned model on UNSEEN songs. Usage: python -m src.evaluate_dj"""
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.dj_model import validate
from src.dj_data import load_rows, fill_names

SIGNOFFS = ("Stay golden", "DJ Nova", "Turn it up", "Trust me")


def run(model_dir, rows, songs):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).eval()
    faithful = voice = 0
    for ex in rows:
        enc = tok(ex["input"], return_tensors="pt")
        out = tok.decode(model.generate(**enc, max_new_tokens=120, num_beams=4)[0], skip_special_tokens=True)
        f = dict(p.split("=", 1) for p in ex["input"][len("dj nova: "):].split(" | "))
        song = next(x for x in songs if x["title"] == ex["title"])
        out = fill_names(out, song)
        faithful += validate(out, song, f["mood_match"] == "yes", f["genre_match"] == "yes") is None
        voice += any(s in out for s in SIGNOFFS)
    n = len(rows)
    print(f"{model_dir:28s} faithful={faithful}/{n} ({faithful/n:.0%})  in-voice={voice}/{n} ({voice/n:.0%})")


if __name__ == "__main__":
    songs = load_rows("data/songs.csv")
    rows = [json.loads(l) for l in open("data/dj/heldout.jsonl")]
    run("google/flan-t5-small", rows, songs)
    run("models/dj_nova", rows, songs)
