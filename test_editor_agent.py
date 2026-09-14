import subprocess
from pathlib import Path

from subtitle_agent import create_subtitles
from edit_timeline_agent import plan_edit_timeline
import editor_agent
from editor_agent import _join_clips, _motion_filter, _run_ffmpeg, edit_video


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
                "shot_id": "shot-001",
                "scene_id": "scene-001",
                "scene_number": 1,
                "start_time": "00:00",
                "end_time": "00:01",
                "motion": "zoom_in",
                "narration": "Raja and Rani enter.",
            }
        ],
        "image_files": [str(image_file)],
        "scene_video_files": [str(scene_video_file)],
        "generated_videos": [{"shot_id": "shot-001", "video_file": str(scene_video_file)}],
        "video_validation_results": [{
            "shot_id": "shot-001", "valid": True, "issues": [],
            "metadata": {"duration_seconds": 1.0},
        }],
        "actual_narration_seconds": 1.0,
        "narration_file": str(narration_file),
        "sfx_files": [str(sfx_file)],
    }
    state["shot_plan"] = [{
        **state["storyboard"][0],
        "source_segment_ids": ["segment-001"],
        "visual_beat_ids": ["beat-001"],
        "estimated_duration_seconds": 1.0,
        "shot_purpose": "Establish Raja and Rani.",
    }]
    state.update(plan_edit_timeline(state))
    state.update(create_subtitles(state))

    result = edit_video(state)
    assert Path(result["mixed_audio_file"]).exists()
    assert Path(result["video_file"]).exists()
    assert state["timeline_valid"] is True
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
    assert "1.05" in _motion_filter("zoom_in", 1)
    assert "setsar=1" in _motion_filter("unknown", 1)
    assert "scale=1920:1080" in _motion_filter("static", 1, 1920, 1080)


def test_approved_rough_cut_is_reused(monkeypatch, tmp_path):
    from PIL import Image

    rough_cut = tmp_path / "rough.mp4"
    rough_cut.write_bytes(b"approved")
    narration = tmp_path / "narration.mp3"
    narration.write_bytes(b"audio")
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    commands = []
    monkeypatch.setattr(editor_agent, "_run_ffmpeg", lambda command: commands.append(command))
    monkeypatch.setattr(editor_agent, "_join_clips", lambda *_args: (_ for _ in ()).throw(AssertionError("rough cut was rebuilt")))

    editor_agent.edit_video({
        "topic": "Approved",
        "output_dir": str(tmp_path),
        "aspect_ratio": "16:9",
        "width": 1920,
        "height": 1080,
        "combined_video_judge_approved": True,
        "rough_cut_file": str(rough_cut),
        "storyboard": [{"scene_number": 1, "start_time": "00:00", "end_time": "00:01"}],
        "image_files": [str(image)],
        "scene_timings": [{"scene_number": 1, "start_seconds": 0, "end_seconds": 1}],
        "narration_file": str(narration),
        "subtitles": [{"start_seconds": 0, "end_seconds": 1, "text": "Approved"}],
    })

    assert str(rough_cut) in commands[-1]
    assert Image.open(tmp_path / "approved" / "video" / "captions" / "caption_001.png").size == (1920, 1080)


def test_ffmpeg_timeout_is_reported(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 300)

    monkeypatch.setattr(editor_agent.subprocess, "run", timeout)
    try:
        _run_ffmpeg(["ffmpeg"])
    except RuntimeError as exc:
        assert "timed out" in str(exc)
        return
    raise AssertionError("FFmpeg timeout should stop the pipeline")


def test_dissolve_keeps_requested_runtime(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    joined = tmp_path / "joined.mp4"
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x568:r=30:d=1.5", "-pix_fmt", "yuv420p", str(first)])
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x568:r=30:d=1", "-pix_fmt", "yuv420p", str(second)])
    _join_clips(
        [first, second],
        [
            {"transition_to_next": {"type": "dissolve", "duration_seconds": 0.5}},
            {"transition_to_next": {"type": "cut", "duration_seconds": 0}},
        ],
        [1, 1],
        joined,
    )
    duration = float(output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(joined)]))
    assert abs(duration - 2) < 0.1


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
    with TemporaryDirectory() as tmp:
        test_dissolve_keeps_requested_runtime(Path(tmp))
    test_motion_filter()
    test_requires_assets()
