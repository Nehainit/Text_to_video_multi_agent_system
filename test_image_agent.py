import base64
from pathlib import Path

import image_agent


PNG_1X1 = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
).decode("ascii")


def test_create_scene_images(tmp_path):
    original_request = image_agent._json_request
    original_headers = image_agent._magnific_headers
    original_sleep = image_agent.time.sleep
    calls = []

    def fake_request(url, payload, headers):
        calls.append((url, payload))
        if url == image_agent.MAGNIFIC_IMAGE_TO_VIDEO_URL:
            return {"data": {"generated": [base64.b64encode(b"fake mp4").decode("ascii")]}}
        return {"data": [{"base64": PNG_1X1}]}

    image_agent._json_request = fake_request
    image_agent._magnific_headers = lambda: {}
    image_agent.time.sleep = lambda seconds: None
    try:
        result = image_agent.create_scene_images(
            {
                "topic": "Raja and Rani",
                "tone": "mystical",
                "duration": "1 minute",
                "language": "English",
                "characters": "Raja, Rani",
                "story": "Raja and Rani enter a glowing forest.",
                "output_dir": str(tmp_path),
                "storyboard": [
                    {
                        "scene_number": 1,
                        "start_time": "00:00",
                        "end_time": "00:12",
                        "visuals": "Raja and Rani in a glowing forest.",
                        "camera": "wide 35mm low angle zoom toward Raja and Rani",
                        "motion": "zoom_in",
                        "narration": "They enter the forest.",
                    }
                ],
            }
        )
    finally:
        image_agent._json_request = original_request
        image_agent._magnific_headers = original_headers
        image_agent.time.sleep = original_sleep

    image_file = Path(result["image_files"][0])
    mood_board_file = Path(result["mood_board_file"])
    scene_video_file = Path(result["scene_video_files"][0])
    reference_files = [Path(path) for path in result["character_reference_files"]]
    assert len(reference_files) == 1
    assert all(path.exists() for path in reference_files)
    assert reference_files[0].name == "characters_reference.png"
    assert image_file.exists()
    assert image_file.name == "scene_01_v001.png"
    assert mood_board_file.exists()
    assert scene_video_file.exists()
    assert scene_video_file.name == "scene_01.mp4"
    assert calls[0][0] == image_agent.MAGNIFIC_TEXT_TO_IMAGE_URL
    assert calls[1][0] == image_agent.MAGNIFIC_TEXT_TO_IMAGE_URL
    assert calls[2][0] == image_agent.MAGNIFIC_TEXT_TO_IMAGE_URL
    assert calls[3][0] == image_agent.MAGNIFIC_IMAGE_TO_VIDEO_URL
    assert "reference_images" not in calls[2][1]
    assert "Clean combined character reference sheet for: Raja, Rani." in calls[0][1]["prompt"]
    assert "Cinematic mood board" in calls[1][1]["prompt"]
    assert "Raja and Rani in a glowing forest." in calls[2][1]["prompt"]
    assert "wide 35mm low angle zoom toward Raja and Rani" in calls[2][1]["prompt"]
    assert "Editor motion to support: zoom_in" in calls[2][1]["prompt"]
    assert "Raja and Rani in a glowing forest." in calls[3][1]["prompt"]
    assert "wide 35mm low angle zoom toward Raja and Rani" in calls[3][1]["prompt"]
    assert "Motion direction: zoom_in" in calls[3][1]["prompt"]


def test_requires_storyboard():
    try:
        image_agent.create_scene_images(
            {
                "topic": "x",
                "tone": "y",
                "duration": "1 minute",
                "language": "English",
                "characters": "Raja",
            }
        )
    except RuntimeError:
        return
    raise AssertionError("image agent should require storyboard")


def test_video_credit_failure_uses_still_images(tmp_path):
    original_request = image_agent._json_request
    original_headers = image_agent._magnific_headers
    original_sleep = image_agent.time.sleep

    def fake_request(url, payload, headers):
        if url == image_agent.MAGNIFIC_IMAGE_TO_VIDEO_URL:
            raise RuntimeError('HTTP 502 {"message":"Error consuming credits"}')
        return {"data": [{"base64": PNG_1X1}]}

    image_agent._json_request = fake_request
    image_agent._magnific_headers = lambda: {}
    image_agent.time.sleep = lambda seconds: None
    try:
        result = image_agent.create_scene_images(
            {
                "topic": "Credit fallback",
                "characters": "Raja",
                "story": "Raja walks through a forest.",
                "tone": "cinematic",
                "output_dir": str(tmp_path),
                "storyboard": [
                    {
                        "scene_number": 1,
                        "visuals": "Raja walks through a forest.",
                        "camera": "wide shot",
                        "motion": "zoom_in",
                    }
                ],
            }
        )
    finally:
        image_agent._json_request = original_request
        image_agent._magnific_headers = original_headers
        image_agent.time.sleep = original_sleep

    assert Path(result["image_files"][0]).exists()
    assert result["scene_video_files"] == []


def test_regenerates_only_scenes_with_feedback(tmp_path):
    original_request = image_agent._json_request
    original_headers = image_agent._magnific_headers
    original_sleep = image_agent.time.sleep
    prompts = []

    def fake_request(url, payload, headers):
        prompts.append(payload["prompt"])
        return {"data": [{"base64": PNG_1X1}]}

    image_agent._json_request = fake_request
    image_agent._magnific_headers = lambda: {}
    image_agent.time.sleep = lambda seconds: None
    state = {
        "topic": "Selective regeneration",
        "tone": "cinematic",
        "characters": "Raja",
        "story": "Raja crosses a forest and reaches a lake.",
        "output_dir": str(tmp_path),
        "storyboard": [
            {"scene_number": 1, "visuals": "Raja in a forest.", "camera": "wide low angle shot", "motion": "pan_left"},
            {"scene_number": 2, "visuals": "Raja at a lake.", "camera": "close eye level shot", "motion": "zoom_in"},
        ],
    }
    try:
        first = image_agent.create_visual_storyboard(state)
        second = image_agent.create_visual_storyboard(
            {**state, **first, "visual_feedback": [{"scene_number": 2, "note": "make the lake brighter"}]}
        )
    finally:
        image_agent._json_request = original_request
        image_agent._magnific_headers = original_headers
        image_agent.time.sleep = original_sleep

    assert first["image_files"][0] == second["image_files"][0]
    assert first["image_files"][1] != second["image_files"][1]
    assert Path(second["image_files"][1]).name == "scene_02_v002.png"
    assert len(prompts) == 5
    assert "Human visual revision request: make the lake brighter" in prompts[-1]


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_create_scene_images(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_video_credit_failure_uses_still_images(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_regenerates_only_scenes_with_feedback(Path(tmp))
    test_requires_storyboard()
