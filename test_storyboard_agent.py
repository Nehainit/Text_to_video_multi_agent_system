import json

import director_agent


BIBLE = {
    "theme": "homecoming",
    "characters": [
        {"character_id": "mira_001", "name": "Mira", "appearance": "runner", "wardrobe": "red coat", "props": []}
    ],
}


def shot(number, purpose, scene="A", duration=5):
    return {
        "shot_id": f"draft-{number}",
        "scene_id": scene,
        "story_purpose": purpose,
        "characters_present": ["mira_001"],
        "duration_seconds": duration,
        "narration": "Mira races through the dark forest toward home." if purpose != "resolution" else "At last, Mira reaches her warm home safely.",
        "visuals": f"Mira in connected shot {number}, wearing the same red coat.",
        "camera": "wide 35mm eye level tracking composition",
        "motion": "pan_right",
        "subject_motion": "Mira runs naturally toward frame right.",
        "continuity_in": "Mira in red coat, frame left, moving right.",
        "continuity_out": "Mira in red coat, frame right, moving right.",
        "transition_to_next": {"type": "cut", "duration_seconds": 0, "audio_bridge": "none"},
        "sfx_prompt": "Footsteps and soft wind.",
    }


class FakeModel:
    def invoke(self, prompt):
        assert "production bible" in prompt.lower()
        opening = shot(1, "opening")
        opening["narration"] = "Mira races through the dark forest"
        resolution = shot(2, "resolution")
        resolution["narration"] = "and reaches her warm home safely."
        return type("Response", (), {"content": json.dumps({"director_plan": [opening, resolution]})})()


def test_director_uses_prompt_count_and_exact_duration():
    original = director_agent.load_model
    director_agent.load_model = lambda _name: FakeModel()
    try:
        result = director_agent.create_director_plan(
            {
                "topic": "A complete film in 2 shots",
                "tone": "cinematic",
                "duration": "10 seconds",
                "language": "English",
                "characters": ["Mira — red coat"],
                "story": "Mira races through the dark forest and reaches her warm home safely.",
                "story_beats": [{"purpose": "opening"}, {"purpose": "turn"}, {"purpose": "resolution"}],
                "production_bible": BIBLE,
            }
        )
    finally:
        director_agent.load_model = original

    assert [item["shot_id"] for item in result["director_plan"]] == ["shot-001", "shot-002"]
    assert sum(item["duration_seconds"] for item in result["director_plan"]) == 10
    assert result["director_plan"][-1]["end_time"] == "00:10"


def test_scenes_and_shots_are_distinct_and_infeasible_count_is_reduced():
    minimum, maximum, shots, scenes, note = director_agent._count_targets("Use 3 scenes and 10 shots", "10 seconds")
    assert (minimum, maximum, shots, scenes) == (1, 5, 5, 3)
    assert "adjusted to 5" in note


def test_duration_normalization_respects_provider_bounds():
    durations = director_agent._fit_durations([{"duration_seconds": 8}, {"duration_seconds": 4}], 10)
    assert sum(durations) == 10
    assert all(2 <= value <= 15 for value in durations)


def test_five_second_one_shot_can_open_and_resolve():
    plan = [shot(1, "opening_resolution")]
    result = director_agent._finalize_plan(plan, {"duration": "5 seconds", "production_bible": BIBLE}, 1, 2, 1, 1)
    assert result[0]["duration_seconds"] == 5
    assert result[0]["end_time"] == "00:05"


def test_shot_count_is_calculated_from_narration_not_hardcoded_duration():
    story = " ".join(f"word{index}" for index in range(45))
    minimum, maximum, shots, scenes, note = director_agent._count_targets("A lion helps a mouse", "30 seconds", story)
    assert (minimum, maximum, shots, scenes) == (2, 15, 5, None)
    assert "45 narration words" in note


def test_director_rejects_character_outside_approved_cast():
    plan = [shot(1, "opening_resolution")]
    plan[0]["characters_present"] = ["stranger_001"]
    try:
        director_agent._finalize_plan(plan, {"duration": "5 seconds", "production_bible": BIBLE}, 1, 2, 1, 1)
    except RuntimeError as exc:
        assert "outside the approved cast" in str(exc)
        return
    raise AssertionError("unapproved characters should fail Director validation")


def test_production_bible_cannot_add_story_characters():
    bible = {
        "theme": "homecoming",
        "visual_style": "painted",
        "palette": ["red"],
        "lighting": "soft",
        "camera_language": "wide",
        "locations": ["home"],
        "characters": [
            BIBLE["characters"][0],
            {"character_id": "extra_001", "name": "Extra", "appearance": "unknown", "wardrobe": "none", "props": []},
        ],
        "continuity_rules": [],
    }
    try:
        director_agent._validate_bible(bible, expected_characters=1)
    except RuntimeError as exc:
        assert "exactly the 1 characters" in str(exc)
        return
    raise AssertionError("production bible should preserve the approved story character count")


if __name__ == "__main__":
    test_director_uses_prompt_count_and_exact_duration()
    test_scenes_and_shots_are_distinct_and_infeasible_count_is_reduced()
    test_duration_normalization_respects_provider_bounds()
    test_five_second_one_shot_can_open_and_resolve()
    test_shot_count_is_calculated_from_narration_not_hardcoded_duration()
    test_director_rejects_character_outside_approved_cast()
    test_production_bible_cannot_add_story_characters()
