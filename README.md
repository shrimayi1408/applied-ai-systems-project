# 🎧 Vibezically + DJ Nova: a Music Recommender with a Fine-Tuned Voice

## Original project (Modules 1–3)
**Music Recommender Simulation ("Vibezically").** A small content-based recommender: it represents 20 songs and a user taste profile as data, scores each song with a weighted rule (mood/genre exact match + closeness on energy, valence, acousticness, danceability), and returns the top 5 with a readable "why" for each pick. Its goal was to make recommendations explainable and to study where a simple scorer goes wrong (see `model_card.md`).

## What this version adds
**DJ Nova**: a `flan-t5-small` model **fine-tuned locally** to explain each recommendation in one consistent persona voice (a warm late-night radio host). It is part of the main flow: every recommendation printed by the CLI is narrated by the model, and its output is validated before the user sees it. Why it matters: the scorer is faithful but robotic; the tuned model makes explanations friendly without letting a small model invent facts.

## Architecture
Diagram: [`diagrams/architecture.mmd`](diagrams/architecture.mmd)

1. **Input**: `data/songs.csv` + a user profile (genre, mood, energy).
2. **Recommender (rule-based)**: `load_songs` → `score_song` → `recommend_songs` pick the top 5 and compute *why*.
3. **DJ Nova**: `derive_facts` turns the match results into structured facts; the fine-tuned model writes a blurb with `[TITLE]`/`[ARTIST]` placeholders; code fills the names in.
4. **Guardrail**: `validate` checks the blurb (names present, no false mood claim, length). On failure, or if the model is missing, a deterministic `plain_blurb` is used instead. Every accept/reject is logged to `dj_nova.log`.
5. **Checks**: `pytest` unit tests, `evaluate_dj.py` (base vs fine-tuned on held-out songs), and a human reading the output and eval table.

The model only controls *tone*. Which songs are chosen, and the facts about them, always come from the scorer.

## How the original recommender scores songs
Real platforms combine collaborative filtering (what similar users played) with content-based filtering (song attributes). This project is content-based only: it sees song attributes and one user's stated taste, so every recommendation traces back to specific features, but it cannot discover surprising tastes the way collaborative filtering can.

Each `Song` has genre, mood, energy, tempo, valence, danceability and acousticness; the user profile holds any of genre, mood and target values. Anything the user leaves out is not scored.

| Feature | Max points | Type |
|---|---|---|
| Mood | 3 | exact match |
| Genre | 1 | exact match |
| Energy | 4 | proximity |
| Valence | 1 | proximity |
| Acousticness | 1 | proximity |
| Danceability | 1 | proximity |
| **Total** | **11** | |

Proximity scoring: `points = max_points × (1 − |song_value − target_value|)`. Ties go to the closer energy. Tempo is in the data but not scored.

## Experiments
- **Weight shift (genre 2 → 1, energy 2 → 4).** Total moved from 10 to 11, so scores are shown as `/ 11` (`MAX_SCORE` is derived from the weights). For `pop/happy/0.8`, Sunrise City stayed #1 (6.96/10 → 7.92/11) and energy-close songs gained ground, so energy now dominates ranking more than genre does.
- **Edge-case profiles** (in `src/main.py`): high energy + sad mood (no sad songs exist → falls back to energy only), unknown genre/mood (`k-pop/ecstatic` → every song scores 0.00 and ranking is arbitrary), and `likes_acoustic` with `acousticness=0.0` (rewards the *least* acoustic songs, a profile/feature mismatch).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.dj_data              # build training data  -> data/dj/*.jsonl
python -m src.train_dj             # fine-tune (~2 min on CPU) -> models/dj_nova
python -m src.main                 # run with DJ Nova   (use --plain for scores only)
python -m src.evaluate_dj          # base vs fine-tuned comparison
pytest                             # unit tests
```
Training is seeded; the first run downloads `google/flan-t5-small` (~300 MB). If `models/dj_nova` is absent, the app still runs using the fallback text and logs a warning.

## Sample interactions
**1. Profile `genre=pop, mood=happy, energy=0.8`**
```
#1  Sunrise City  -  Neon Echo      Score: 7.92 / 11
    DJ Nova: Dropping the needle on Sunrise City by Neon Echo. A high-voltage pop track.
    Pure happy, exactly your speed. Proper pop, just how you like it. The energy sits
    right where you wanted it. This is DJ Nova, keeping the night warm.
```
**2. Profile `energy=0.9, mood=sad` (no sad songs exist)**
```
#1  Storm Runner  -  Voltline      Score: 3.96 / 11
    DJ Nova: Don't touch that knob, it's Storm Runner by Voltline. A full-throttle rock
    track. It's not quite the intense you ordered, but hear me out. The energy sits right
    where you wanted it. Turn it up, you've earned it.
```
**3. Profile `genre=k-pop, mood=ecstatic` (unknown to the catalog)**
```
#1  4Me 4Me  -  Malcolm Todd       Score: 0.00 / 11
    DJ Nova: ... It's not quite the chill you ordered, but hear me out. Energy's close
    enough to feel right. Turn it up, you've earned it.
```
(Interaction 3 exposes a flaw: with no energy target, the "energy" sentence is not grounded. See Limitations in `model_card.md`.)

## Design decisions and trade-offs
- **Fine-tune a small local model instead of prompting a big one.** Reproducible with no API key and it is a genuine specialized model, but 640 synthetic examples only teach tone, not language skill.
- **Template-generated training data.** Fast, consistent voice, and the facts are correct by construction. Cost: the model can only produce variations of the templates.
- **Placeholders for names.** The first version let the model copy titles and artists itself and garbled unseen names ("Dust Road Sunday" → "Storm Road Sunday"; 50% faithful on held-out songs). Substituting names in code fixed it (100%).
- **Guardrail + fallback over trusting the model.** Costs a few lines of code; makes the system safe if the model is missing or misbehaves.
- **Scorer stays rule-based.** Keeps recommendations explainable.

## Testing summary
Held-out evaluation (`python -m src.evaluate_dj`, 40 blurbs for 4 songs never seen in training):

| Model | Faithful (passes `validate`) | In DJ voice |
|---|---|---|
| `google/flan-t5-small` (base) | 0/40 (0%) | 0/40 (0%) |
| `models/dj_nova` (fine-tuned, name-copy by model) | 20/40 (50%) | 40/40 (100%) |
| `models/dj_nova` (fine-tuned, placeholders) | 40/40 (100%) | 40/40 (100%) |

| Test input | Criteria | Result |
|---|---|---|
| Garbled title in blurb | Rejected by `validate` | Pass |
| Empty / 700-char blurb | Rejected | Pass |
| False "that happy vibe" claim on non-match | Rejected | Pass |
| Missing model directory | App falls back, no crash | Pass |
| Profile with no energy target | Blurb should not claim an energy fit | **Fail** (known, documented) |

`pytest`: 7 passed. Learned: the metric that mattered was faithfulness, not fluency, and the weakness (copying unseen names) only showed up on held-out data. The "voice" metric is a sign-off phrase check, which is a weak proxy for tone quality.

## Limitations (short version)
Tiny 20-song catalog with no sad mood; no lyrics or language understanding; mood over-weighted relative to genre; the voice model only rephrases and can state an ungrounded energy claim when no energy target is given. Details in `model_card.md`.

## Reflection
This project showed me that fine-tuning is the easy part; deciding what the model is *allowed* to do is the real design work. Splitting responsibilities (rules decide, model phrases, code validates) made the system more reliable than letting the model do everything. Ethics, limitations and AI-collaboration notes are in [`model_card.md`](model_card.md).
