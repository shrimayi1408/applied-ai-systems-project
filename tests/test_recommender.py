from src.recommender import Song, UserProfile, Recommender

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---- DJ Nova guardrail / scoring tests ----
from src.recommender import score_song, MAX_SCORE
from src.dj_model import derive_facts, validate, plain_blurb

SONG = {"title": "Sunrise City", "artist": "Neon Echo", "genre": "pop", "mood": "happy", "energy": 0.82}


def test_score_never_exceeds_max_and_perfect_match_is_high():
    full = {**SONG, "valence": 0.8, "acousticness": 0.2, "danceability": 0.8}
    score, _ = score_song({"genre": "pop", "mood": "happy", "energy": 0.82}, full)
    assert 0 < score <= MAX_SCORE


def test_derive_facts():
    assert derive_facts({"mood": "happy", "genre": "rock", "energy": 0.8}, SONG) == (True, False, "tight")
    assert derive_facts({"mood": "sad", "energy": 0.2}, SONG)[2] == "far"


def test_validate_accepts_faithful_blurb():
    assert validate("Sunrise City by Neon Echo. Pure happy.", SONG, True, False) is None


def test_validate_rejects_garbled_name_and_false_claims():
    assert validate("Sunrize City by Neon Echo.", SONG, True, False) is not None
    assert validate("", SONG, True, False) is not None
    assert validate("Sunrise City by Neon Echo. That happy vibe?", SONG, False, False) is not None
    assert validate("x" * 700, SONG, True, False) is not None


def test_missing_model_falls_back_gracefully():
    from src.dj_model import DJNova
    dj = DJNova(model_dir="does/not/exist")
    assert not dj.ready
    text = dj.narrate({"mood": "happy", "genre": "pop", "energy": 0.8}, SONG, 1)
    assert text == plain_blurb(SONG, True, True, "tight")
