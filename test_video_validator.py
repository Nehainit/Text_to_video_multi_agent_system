from pathlib import Path
import shutil
import subprocess

import pytest
import video_validator


def _probe(*, duration="4.0", width=1280, height=720, fps="24/1", frames="96", audio=False):
    streams = [{
        "codec_type": "video", "codec_name": "h264", "duration": duration,
        "width": width, "height": height, "avg_frame_rate": fps, "r_frame_rate": fps,
        "nb_frames": frames,
    }]
    if audio:
        streams.append({"codec_type": "audio", "codec_name": "aac"})
    return {"streams": streams, "format": {"duration": duration}}


def _file(tmp_path: Path, content=b"video") -> Path:
    path = tmp_path / "shot.mp4"
    path.write_bytes(content)
    return path


def _ready(monkeypatch, probe=None, decodes=True):
    monkeypatch.setattr(video_validator, "_probe", lambda _path: probe or _probe())
    monkeypatch.setattr(video_validator, "_decodes", lambda _path: decodes)


def test_valid_video_passes(monkeypatch, tmp_path):
    _ready(monkeypatch)
    result = video_validator.validate_video("shot-001", str(_file(tmp_path)), requested_duration_seconds=4, expected_aspect_ratio=16 / 9)
    assert result == {
        "shot_id": "shot-001", "valid": True, "issues": [],
        "metadata": {
            "duration_seconds": 4.0, "width": 1280, "height": 720, "aspect_ratio": 16 / 9,
            "fps": 24.0, "frame_count": 96, "video_codec": "h264", "has_video_stream": True,
            "has_audio_stream": False, "file_size_bytes": 5,
        },
    }


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools are required")
def test_real_ffmpeg_video_passes(tmp_path):
    video = tmp_path / "real.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x180:r=24",
        "-t", "1", "-c:v", "mpeg4", "-an", str(video),
    ], check=True)
    result = video_validator.validate_video(
        "shot-001", str(video), requested_duration_seconds=1,
        expected_width=320, expected_height=180, expected_aspect_ratio=16 / 9,
    )
    assert result["valid"] is True
    assert result["metadata"]["frame_count"] == 24


def test_missing_file_fails(tmp_path):
    result = video_validator.validate_video("shot-001", str(tmp_path / "missing.mp4"))
    assert result["valid"] is False
    assert result["issues"][0]["type"] == "video_file_missing"


def test_empty_file_fails(tmp_path):
    result = video_validator.validate_video("shot-001", str(_file(tmp_path, b"")))
    assert result["valid"] is False
    assert result["issues"][0]["type"] == "video_file_empty"


def test_corrupt_video_fails_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(video_validator, "_probe", lambda _path: (_ for _ in ()).throw(RuntimeError("invalid data")))
    result = video_validator.validate_video("shot-001", str(_file(tmp_path)))
    assert result["valid"] is False
    assert result["issues"][0]["type"] == "video_probe_failed"


def test_undecodable_video_fails(monkeypatch, tmp_path):
    _ready(monkeypatch, decodes=False)
    result = video_validator.validate_video("shot-001", str(_file(tmp_path)))
    assert "video_decode_failed" in [issue["type"] for issue in result["issues"]]


def test_duration_outside_tolerance_fails(monkeypatch, tmp_path):
    _ready(monkeypatch, _probe(duration="4.51"))
    result = video_validator.validate_video("shot-001", str(_file(tmp_path)), requested_duration_seconds=4)
    assert "duration_mismatch" in [issue["type"] for issue in result["issues"]]


def test_duration_inside_tolerance_passes(monkeypatch, tmp_path):
    _ready(monkeypatch, _probe(duration="4.5"))
    assert video_validator.validate_video("shot-001", str(_file(tmp_path)), requested_duration_seconds=4)["valid"] is True


def test_generated_provider_duration_is_the_validation_contract(monkeypatch, tmp_path):
    _ready(monkeypatch, _probe(duration="5"))
    result = video_validator.validate_generated_videos({
        "shot_plan": [{"shot_id": "shot-001"}],
        "generated_videos": [{
            "shot_id": "shot-001",
            "video_file": str(_file(tmp_path)),
            "requested_duration_seconds": 4,
            "generated_duration_seconds": 5,
            "aspect_ratio": "16:9",
        }],
    })

    assert result["video_validation_results"][0]["valid"] is True
    assert result["video_retry_shots"] == []


def test_wrong_expected_resolution_fails(monkeypatch, tmp_path):
    _ready(monkeypatch)
    result = video_validator.validate_video("shot-001", str(_file(tmp_path)), expected_width=1920, expected_height=1080)
    assert [issue["type"] for issue in result["issues"]] == ["width_mismatch", "height_mismatch"]


def test_missing_audio_passes_when_optional(monkeypatch, tmp_path):
    _ready(monkeypatch)
    monkeypatch.setitem(video_validator.VIDEO_VALIDATION_CONFIG, "require_audio", False)
    assert video_validator.validate_video("shot-001", str(_file(tmp_path)))["valid"] is True


def test_missing_audio_fails_when_required(monkeypatch, tmp_path):
    _ready(monkeypatch)
    monkeypatch.setitem(video_validator.VIDEO_VALIDATION_CONFIG, "require_audio", True)
    result = video_validator.validate_video("shot-001", str(_file(tmp_path)))
    assert "audio_stream_missing" in [issue["type"] for issue in result["issues"]]


def test_validation_retries_only_failed_shot(monkeypatch):
    monkeypatch.setattr(video_validator, "validate_video", lambda shot_id, *_args, **_kwargs: {
        "shot_id": shot_id, "valid": shot_id == "shot-001", "issues": [] if shot_id == "shot-001" else [{"type": "video_decode_failed", "description": "bad"}],
        "metadata": video_validator._metadata(5),
    })
    result = video_validator.validate_generated_videos({
        "shot_plan": [{"shot_id": "shot-001"}, {"shot_id": "shot-002"}],
        "generated_videos": [
            {"shot_id": "shot-001", "video_file": "one.mp4", "requested_duration_seconds": 4},
            {"shot_id": "shot-002", "video_file": "two.mp4", "requested_duration_seconds": 4},
        ],
    })
    assert result["video_retry_shots"] == ["shot-002"]
    assert result["video_validation_retry_counts"] == {"shot-002": 1}
    assert result["scene_video_files"] == ["one.mp4", None]


def test_validation_marks_shot_exhausted_after_max_retries(monkeypatch):
    monkeypatch.setattr(video_validator, "validate_video", lambda shot_id, *_args, **_kwargs: {
        "shot_id": shot_id, "valid": False,
        "issues": [{"type": "video_decode_failed", "description": "bad"}],
        "metadata": video_validator._metadata(5),
    })
    result = video_validator.validate_generated_videos({
        "shot_plan": [{"shot_id": "shot-001"}],
        "generated_videos": [{"shot_id": "shot-001", "video_file": "one.mp4", "requested_duration_seconds": 4}],
        "video_validation_results": [{"shot_id": "shot-001", "valid": False, "issues": [], "metadata": video_validator._metadata()}],
        "video_validation_retry_counts": {"shot-001": 2},
        "video_retry_shots": ["shot-001"],
    })
    assert result["video_retry_shots"] == []
    assert result["video_validation_exhausted_shots"] == ["shot-001"]
    assert result["pipeline_status"] == "video_validation_failed"
