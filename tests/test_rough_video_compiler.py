import shutil
import subprocess
from pathlib import Path

import pytest

from video_automation.agents import rough_video_compiler as compiler
from video_automation.agents.rough_video_compiler import build_ffmpeg_command, compile_rough_video


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg tools are required",
)


def _run(command):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("rough-cut")
    first, second, narration = root / "first.mp4", root / "second.mp4", root / "narration.wav"
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=30:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(first)])
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=640x360:r=24:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(second)])
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2", str(narration)])
    return root, first, second, narration


def _state(media):
    root, first, second, narration = media
    return {
        "output_dir": str(root / "output"),
        "aspect_ratio": "16:9",
        "timeline_valid": True,
        "timeline_duration_seconds": 2.0,
        "narration_duration_seconds": 2.0,
        "narration_file": str(narration),
        "edit_timeline": [
            {
                "timeline_item_id": "edit-001", "shot_id": "shot-001", "scene_id": "scene-001",
                "video_file": str(first), "source_in": 0.25, "source_out": 1.25,
                "timeline_start": 0.0, "timeline_end": 1.0,
                "narration_segment_ids": ["segment-001"], "transition_in": "cut", "transition_out": "cut",
            },
            {
                "timeline_item_id": "edit-002", "shot_id": "shot-002", "scene_id": "scene-002",
                "video_file": str(second), "source_in": 0.5, "source_out": 1.5,
                "timeline_start": 1.0, "timeline_end": 2.0,
                "narration_segment_ids": ["segment-002"], "transition_in": "cut", "transition_out": "cut",
            },
        ],
        "video_validation_results": [
            {"shot_id": "shot-001", "valid": True, "metadata": {"duration_seconds": 2.0}},
            {"shot_id": "shot-002", "valid": True, "metadata": {"duration_seconds": 2.0}},
        ],
    }


@pytest.fixture(scope="module")
def compiled(media):
    state = _state(media)
    return state, compile_rough_video(state)


def _types(result):
    return [issue["type"] for issue in result["compilation_issues"]]


def test_valid_timeline_compiles(compiled):
    _, result = compiled
    assert result["compilation_status"] == "success"
    assert Path(result["rough_cut_file"]).is_file()
    assert result["sync_valid"] is True


def test_timeline_missing_fails(media):
    state = _state(media)
    state.pop("edit_timeline")
    assert _types(compile_rough_video(state)) == ["timeline_missing"]


def test_empty_timeline_fails(media):
    state = _state(media)
    state["edit_timeline"] = []
    assert "timeline_empty" in _types(compile_rough_video(state))


def test_missing_narration_fails(media):
    state = _state(media)
    state["narration_file"] = str(Path(state["narration_file"]).with_name("missing.wav"))
    assert "narration_file_missing" in _types(compile_rough_video(state))


def test_missing_source_clip_fails_with_shot_id(media):
    state = _state(media)
    state["edit_timeline"][1]["video_file"] = "missing.mp4"
    result = compile_rough_video(state)
    issue = next(issue for issue in result["compilation_issues"] if issue["type"] == "video_file_missing")
    assert issue["shot_id"] == "shot-002"
    assert result["video_retry_shots"] == ["shot-002"]


def test_invalid_source_range_fails(media):
    state = _state(media)
    state["edit_timeline"][0]["source_out"] = 0.25
    assert "invalid_source_range" in _types(compile_rough_video(state))


def test_source_range_beyond_clip_fails(media):
    state = _state(media)
    state["edit_timeline"][1]["source_out"] = 3.0
    state["edit_timeline"][1]["timeline_end"] = 3.5
    state["timeline_duration_seconds"] = 3.5
    assert "source_range_exceeds_clip" in _types(compile_rough_video(state))


def test_timeline_gap_is_detected(media):
    state = _state(media)
    state["edit_timeline"][1]["timeline_start"] = 1.2
    state["edit_timeline"][1]["timeline_end"] = 2.2
    assert "timeline_gap" in _types(compile_rough_video(state))


def test_invalid_overlap_is_detected(media):
    state = _state(media)
    state["edit_timeline"][1]["timeline_start"] = 0.8
    state["edit_timeline"][1]["timeline_end"] = 1.8
    assert "timeline_overlap" in _types(compile_rough_video(state))


def test_narration_timeline_duration_mismatch_fails(media):
    state = _state(media)
    state["timeline_duration_seconds"] = 3.0
    assert "timeline_narration_duration_mismatch" in _types(compile_rough_video(state))


def _pixel(video_file, second):
    return subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", str(second), "-i", video_file,
        "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])[:3]


def test_clips_are_concatenated_in_timeline_order(compiled):
    _, result = compiled
    first, second = _pixel(result["rough_cut_file"], 0.25), _pixel(result["rough_cut_file"], 1.25)
    assert first[0] > first[2]
    assert second[2] > second[0]


def test_command_uses_exact_source_trims(media):
    state = _state(media)
    command = build_ffmpeg_command(
        state["edit_timeline"], state["narration_file"], "rough.mp4",
        width=1280, height=720, fps=24, duration=2,
    )
    filters = command[command.index("-filter_complex") + 1]
    assert "trim=start=0.250000:end=1.250000" in filters
    assert "trim=start=0.500000:end=1.500000" in filters


def test_output_contains_video_and_narration_streams(compiled):
    _, result = compiled
    streams = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
        result["rough_cut_file"],
    ], text=True).splitlines()
    assert "video" in streams
    assert "audio" in streams


def test_output_duration_and_normalization_are_correct(compiled):
    _, result = compiled
    assert abs(result["duration_seconds"] - 2.0) <= 0.25
    assert (result["width"], result["height"], result["fps"]) == (1920, 1080, 30.0)


def test_shot_timestamp_mapping_is_preserved(compiled):
    _, result = compiled
    assert result["shot_timestamps"] == [
        {
            "timeline_item_id": "edit-001", "shot_id": "shot-001", "scene_id": "scene-001",
            "start_sec": 0.0, "end_sec": 1.0, "narration_segment_ids": ["segment-001"],
        },
        {
            "timeline_item_id": "edit-002", "shot_id": "shot-002", "scene_id": "scene-002",
            "start_sec": 1.0, "end_sec": 2.0, "narration_segment_ids": ["segment-002"],
        },
    ]


def test_ffmpeg_failure_exhausts_configured_retries(monkeypatch, media):
    state = _state(media)
    state["rough_cut_retry_count"] = compiler.VIDEO_COMPILATION_CONFIG["max_retries"]
    monkeypatch.setattr(compiler, "_run_ffmpeg", lambda _command: (_ for _ in ()).throw(RuntimeError("failed")))
    result = compile_rough_video(state)
    assert result["rough_cut_retry_exhausted"] is True
    assert result["pipeline_status"] == "rough_cut_failed"
