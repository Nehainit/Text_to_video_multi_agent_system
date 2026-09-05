import storyboard_agent


class FakeModel:
    def invoke(self, prompt):
        assert "Raja and Rani escaped the devil." in prompt
        assert "use warmer lighting" in prompt
        scenes = []
        for scene_number in range(1, 4):
            scenes.append(
                (
                    f'{{"scene_number": {scene_number}, "start_time": "00:{(scene_number - 1) * 10:02d}", '
                    f'"end_time": "00:{scene_number * 10:02d}", '
                    f'"visuals": "Scene {scene_number} visual.", '
                    f'"camera": "wide 35mm low angle zoom toward Raja and Rani", "motion": "zoom_in", '
                    f'"narration": "Scene {scene_number} narration."}}'
                )
            )
        return type(
            "Response",
            (),
            {
                "content": '{"storyboard": [' + ",".join(scenes) + "]}"
            },
        )()


def test_create_storyboard():
    original = storyboard_agent.load_model
    storyboard_agent.load_model = lambda agent_name: FakeModel()
    try:
        result = storyboard_agent.create_storyboard(
            {
                "topic": "x",
                "tone": "mystical",
                "duration": "30 seconds",
                "language": "English",
                "characters": "Raja, Rani, Devil",
                "story": "Raja and Rani escaped the devil.",
                "storyboard_review_note": "use warmer lighting",
            }
        )
    finally:
        storyboard_agent.load_model = original

    assert len(result["storyboard"]) == 3
    assert result["storyboard"][0]["camera"] == "wide 35mm low angle zoom toward Raja and Rani"
    assert result["storyboard"][0]["motion"] == "zoom_in"


def test_rejects_bad_motion():
    try:
        storyboard_agent._validate_storyboard(
            [
                {
                    "scene_number": 1,
                    "start_time": "00:00",
                    "end_time": "00:10",
                    "visuals": "Scene visual.",
                    "camera": "wide 35mm low angle shot",
                    "motion": "spin",
                    "narration": "Scene narration.",
                }
            ],
            1,
            "10 seconds",
        )
    except RuntimeError:
        return
    raise AssertionError("storyboard should reject unsupported motion")


def test_rejects_weak_camera_plan():
    try:
        storyboard_agent._validate_storyboard(
            [
                {
                    "scene_number": 1,
                    "start_time": "00:00",
                    "end_time": "00:10",
                    "visuals": "Scene visual.",
                    "camera": "wide",
                    "motion": "static",
                    "narration": "Scene narration.",
                }
            ],
            1,
            "10 seconds",
        )
    except RuntimeError:
        return
    raise AssertionError("storyboard should reject weak camera plans")


def test_requires_story():
    try:
        storyboard_agent.create_storyboard(
            {
                "topic": "x",
                "tone": "mystical",
                "duration": "1 minute",
                "language": "English",
                "characters": "Raja, Rani, Devil",
            }
        )
    except RuntimeError:
        return
    raise AssertionError("storyboard should require a story")


if __name__ == "__main__":
    test_create_storyboard()
    test_rejects_bad_motion()
    test_rejects_weak_camera_plan()
    test_requires_story()
