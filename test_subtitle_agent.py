import json
from pathlib import Path

from subtitle_agent import create_subtitles


def test_create_subtitles(tmp_path):
    result = create_subtitles(
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
                    "narration": "Raja and Rani enter the forest.",
                },
                {
                    "scene_number": 2,
                    "start_time": "00:12",
                    "end_time": "00:24",
                    "narration": "The devil appears from the mist.",
                },
            ],
        }
    )

    subtitle_file = Path(result["subtitle_file"])
    assert subtitle_file.exists()
    assert "00:00:00,000 --> 00:00:12,000" in subtitle_file.read_text(encoding="utf-8")
    assert result["subtitles"][1]["text"] == "The devil appears from the mist."


def test_requires_storyboard():
    try:
        create_subtitles(
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
    raise AssertionError("subtitles should require storyboard")


def test_word_level_subtitles(tmp_path):
    alignment_file = tmp_path / "alignment.json"
    text = "Raja enters the forest"
    alignment_file.write_text(
        json.dumps(
            {
                "text": text,
                "alignment": {
                    "characters": list(text),
                    "character_start_times_seconds": [index * 0.1 for index in range(len(text))],
                    "character_end_times_seconds": [(index + 1) * 0.1 for index in range(len(text))],
                },
                "scene_timings": [{"scene_number": 1, "start_seconds": 0.0, "end_seconds": 2.2}],
            }
        ),
        encoding="utf-8",
    )
    result = create_subtitles(
        {
            "topic": "Raja and Rani",
            "tone": "mystical",
            "duration": "1 minute",
            "language": "English",
            "characters": "Raja, Rani",
            "output_dir": str(tmp_path),
            "storyboard": [{"scene_number": 1, "start_time": "00:00", "end_time": "00:03", "narration": text}],
            "narration_alignment_file": str(alignment_file),
        }
    )

    assert result["subtitles"][0]["text"] == "Raja enters the forest"
    assert result["subtitles"][0]["end_seconds"] == 2.2


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_create_subtitles(Path(tmp))
        test_word_level_subtitles(Path(tmp))
    test_requires_storyboard()
