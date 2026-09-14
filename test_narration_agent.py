import base64
import os
from pathlib import Path

import narration_agent


def alignment_for(text):
    chars = list(text)
    starts = [index * 0.1 for index in range(len(chars))]
    ends = [(index + 1) * 0.1 for index in range(len(chars))]
    return {
        "characters": chars,
        "character_start_times_seconds": starts,
        "character_end_times_seconds": ends,
    }


def test_create_narration(tmp_path):
    original_request = narration_agent._json_request
    original_headers = narration_agent._elevenlabs_headers
    original_model = os.environ.pop("ELEVENLABS_TTS_MODEL", None)
    calls = []

    def fake_request(url, payload, headers):
        calls.append((url, payload))
        return {"audio_base64": base64.b64encode(b"fake mp3").decode("ascii"), "alignment": alignment_for(payload["text"])}

    narration_agent._json_request = fake_request
    narration_agent._elevenlabs_headers = lambda: {}
    try:
        result = narration_agent.create_narration(
            {
                "topic": "Raja and Rani",
                "tone": "mystical",
                "duration": "1 minute",
                "language": "English",
                "characters": "Raja, Rani",
                "output_dir": str(tmp_path),
                "story": "Fallback story.",
                    "narration_segments": [
                        {"segment_id": "segment-001", "parent_beat_ids": ["beat-001"], "text": "Raja enters the forest."},
                        {"segment_id": "segment-002", "parent_beat_ids": ["beat-002"], "text": "Rani follows the golden light."},
                    ],
                    "storyboard": [
                        {"shot_id": "shot-001", "scene_number": 1, "start_time": "00:00", "end_time": "00:30", "narration": "Raja enters the forest."},
                        {"shot_id": "shot-002", "scene_number": 2, "start_time": "00:30", "end_time": "01:00", "narration": "Rani follows the golden light."},
                    ],
                    "director_plan": [{"shot_id": "shot-001"}, {"shot_id": "shot-002"}],
            }
        )
    finally:
        narration_agent._json_request = original_request
        narration_agent._elevenlabs_headers = original_headers
        if original_model is not None:
            os.environ["ELEVENLABS_TTS_MODEL"] = original_model

    narration_file = Path(result["narration_file"])
    assert narration_file.exists()
    assert narration_file.read_bytes() == b"fake mp3"
    assert Path(result["narration_alignment_file"]).exists()
    assert calls[0][0] == (
        "https://api.elevenlabs.io/v1/text-to-speech/"
        "JBFqnCBsd6RMkjVDRZzb/with-timestamps?output_format=mp3_44100_128"
    )
    assert calls[0][1]["text"] == "Raja enters the forest.\nRani follows the golden light."
    assert calls[0][1]["model_id"] == "eleven_flash_v2_5"
    assert result["actual_narration_seconds"] > 0
    assert result["narration_segment_timings"] == result["scene_timings"]
    assert result["scene_timings"][1]["segment_id"] == "segment-002"
    assert round(result["scene_timings"][1]["start_seconds"], 1) == 2.4


def test_long_voiceover_requests_narration_revision(tmp_path):
    original_request = narration_agent._json_request
    original_headers = narration_agent._elevenlabs_headers

    def fake_request(_url, payload, _headers):
        alignment = alignment_for(payload["text"])
        alignment["character_end_times_seconds"][-1] = 12
        return {"audio_base64": base64.b64encode(b"fake mp3").decode("ascii"), "alignment": alignment}

    narration_agent._json_request = fake_request
    narration_agent._elevenlabs_headers = lambda: {}
    try:
        result = narration_agent.create_narration(
            {
                "topic": "short",
                "duration": "10 seconds",
                "output_dir": str(tmp_path),
                "storyboard": [{"shot_id": "shot-001", "scene_number": 1, "start_time": "00:00", "end_time": "00:10", "narration": "A complete short story."}],
            }
        )
    finally:
        narration_agent._json_request = original_request
        narration_agent._elevenlabs_headers = original_headers
    assert "12.0s" in result["narration_feedback"]


def test_requires_story_or_storyboard():
    try:
        narration_agent.create_narration(
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
    raise AssertionError("narration should require story or storyboard")


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_create_narration(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_long_voiceover_requests_narration_revision(Path(tmp))
    test_requires_story_or_storyboard()
