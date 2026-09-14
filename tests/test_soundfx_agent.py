from pathlib import Path

from video_automation.agents import soundfx_agent


def test_create_soundfx(tmp_path):
    original_request = soundfx_agent._bytes_request
    original_headers = soundfx_agent._elevenlabs_headers
    calls = []

    def fake_request(url, payload, headers):
        calls.append(payload)
        return b"fake sfx"

    soundfx_agent._bytes_request = fake_request
    soundfx_agent._elevenlabs_headers = lambda: {}
    try:
        result = soundfx_agent.create_soundfx(
            {
                "topic": "Raja and Rani",
                "tone": "mystical",
                "duration": "1 minute",
                "language": "English",
                "characters": "Raja, Rani",
                "output_dir": str(tmp_path),
                "storyboard": [
                    {
                        "scene_number": 1,
                        "start_time": "00:00",
                        "end_time": "00:12",
                        "visuals": "A glowing forest.",
                        "camera": "wide",
                        "narration": "Raja enters.",
                    }
                ],
                "scene_timings": [{"scene_number": 1, "start_seconds": 0.0, "end_seconds": 3.5}],
            }
        )
    finally:
        soundfx_agent._bytes_request = original_request
        soundfx_agent._elevenlabs_headers = original_headers

    sfx_file = Path(result["sfx_files"][0])
    assert sfx_file.exists()
    assert sfx_file.read_bytes() == b"fake sfx"
    assert calls[0]["duration_seconds"] == 3.5
    assert "A glowing forest" in calls[0]["text"]


def test_requires_storyboard():
    try:
        soundfx_agent.create_soundfx(
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
    raise AssertionError("soundfx should require storyboard")


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_create_soundfx(Path(tmp))
    test_requires_storyboard()
