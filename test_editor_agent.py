import subprocess
from pathlib import Path

from subtitle_agent import create_subtitles
from editor_agent import _motion_filter, edit_video


def run(command):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def output(command):
    return subprocess.check_output(command, text=True).strip()


def test_edit_video(tmp_path):
    image_file = tmp_path / "scene_01.png"
    scene_video_file = tmp_path / "scene_video_01.mp4"
    narration_file = tmp_path / "narration.mp3"
    sfx_file = tmp_path / "sfx_scene_01.mp3"
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x568:d=1", "-frames:v", "1", str(image_file)])
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x568:d=1", "-t", "1", "-pix_fmt", "yuv420p", str(scene_video_file)])
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", str(narration_file)])
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", str(sfx_file)])

    state = {
        "topic": "Raja and Rani",
        "tone": "mystical",
        "duration": "1 second",
        "language": "English",
        "characters": "Raja, Rani",
        "output_dir": str(tmp_path),
        "storyboard": [
            {
                "scene_number": 1,
                "start_time": "00:00",
                "end_time": "00:01",
                "motion": "zoom_in",
                "narration": "Raja and Rani enter.",
            }
        ],
        "image_files": [str(image_file)],
        "scene_video_files": [str(scene_video_file)],
        "narration_file": str(narration_file),
        "sfx_files": [str(sfx_file)],
    }
    state.update(create_subtitles(state))

    result = edit_video(state)
    assert Path(result["mixed_audio_file"]).exists()
    assert Path(result["video_file"]).exists()
    streams = output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            result["video_file"],
        ]
    )
    assert streams == ""


def test_motion_filter():
    assert "zoompan" in _motion_filter("zoom_in", 1)
    assert "setsar=1" in _motion_filter("unknown", 1)


def test_requires_assets():
    try:
        edit_video(
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
    raise AssertionError("editor agent should require generated assets")


def test_rejects_count_mismatch(tmp_path):
    image_file = tmp_path / "scene_01.png"
    narration_file = tmp_path / "narration.mp3"
    sfx_file = tmp_path / "sfx_scene_01.mp3"
    image_file.write_bytes(b"x")
    narration_file.write_bytes(b"x")
    sfx_file.write_bytes(b"x")
    try:
        edit_video(
            {
                "topic": "x",
                "tone": "y",
                "duration": "1 minute",
                "language": "English",
                "characters": "Raja",
                "output_dir": str(tmp_path),
                "storyboard": [
                    {"scene_number": 1, "start_time": "00:00", "end_time": "00:01", "motion": "static"},
                    {"scene_number": 2, "start_time": "00:01", "end_time": "00:02", "motion": "static"},
                ],
                "image_files": [str(image_file)],
                "narration_file": str(narration_file),
                "sfx_files": [str(sfx_file)],
                "subtitles": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "x"}],
            }
        )
    except RuntimeError as exc:
        assert "asset counts" in str(exc)
        return
    raise AssertionError("editor agent should reject mismatched assets")


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_edit_video(Path(tmp))
        test_rejects_count_mismatch(Path(tmp))
    test_motion_filter()
    test_requires_assets()
