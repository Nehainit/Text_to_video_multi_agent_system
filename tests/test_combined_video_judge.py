import json
import subprocess
from types import SimpleNamespace

import pytest
from langgraph.graph import END

from video_automation import agent_graph
from video_automation.agents import combined_video_judge as judge


METRICS = (
    "visual_fidelity", "character_consistency", "motion_quality", "temporal_consistency",
    "shot_alignment", "narration_alignment", "story_faithfulness", "emotional_intent",
    "pacing_and_editing", "overall_coherence",
)


def report(*, source="none", target="none", issue=None, approved=True):
    scores = {metric: 9 for metric in METRICS}
    if not approved:
        scores["temporal_consistency"] = 6
    return {
        "approved": approved,
        "overall_score": 8.8 if approved else 7.4,
        "scores": scores,
        "issues": [] if issue is None else [{
            "type": "temporal_flicker",
            "severity": "major",
            "start_sec": 2.2,
            "end_sec": 2.8,
            "description": "The image flickers during the second shot.",
            "failure_source": source,
            "retry_target": target,
            "correction": "Regenerate this shot without changing the approved inputs.",
        }],
        "primary_failure_source": source,
        "retry_target": target,
        "refinement_needed": not approved,
        "summary": "Suitable to continue." if approved else "One shot needs correction.",
    }


def timestamps():
    return [
        {"shot_id": "shot-001", "start_sec": 0.0, "end_sec": 2.0},
        {"shot_id": "shot-002", "start_sec": 2.0, "end_sec": 4.0},
    ]


def state(tmp_path):
    rough_cut = tmp_path / "rough_cut.mp4"
    rough_cut.write_bytes(b"video")
    return {
        "rough_cut_file": str(rough_cut),
        "duration_seconds": 4.0,
        "shot_timestamps": timestamps(),
        "shot_plan": [{"shot_id": "shot-001"}, {"shot_id": "shot-002"}],
        "shot_image_qa_results": [
            {"shot_id": "shot-001", "approved": True, "animation_ready": True},
            {"shot_id": "shot-002", "approved": True, "animation_ready": True},
        ],
    }


def mock_frames(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    monkeypatch.setattr(judge, "extract_video_frames", lambda *_args: ([str(frame)], [0.0]))


def test_valid_structured_output_passes_schema_validation():
    result = judge.validate_combined_judge_report(report(), 4.0, timestamps())
    assert result["approved"] is True
    assert result["problem_shot_ids"] == []


def test_invalid_score_range_fails():
    value = report()
    value["scores"]["visual_fidelity"] = 11
    with pytest.raises(RuntimeError, match="invalid structured output"):
        judge.validate_combined_judge_report(value, 4.0, timestamps())


def test_invalid_retry_target_fails():
    value = report(source="video_generation", target="motion_planner", issue=True, approved=False)
    with pytest.raises(RuntimeError, match="invalid structured output"):
        judge.validate_combined_judge_report(value, 4.0, timestamps())


def test_timestamp_maps_to_correct_shot():
    assert judge.map_timestamp_to_shots(2.2, 2.8, timestamps()) == ["shot-002"]


def test_range_overlapping_two_shots_maps_to_both():
    assert judge.map_timestamp_to_shots(1.8, 2.2, timestamps()) == ["shot-001", "shot-002"]


@pytest.mark.parametrize(("state_value", "expected"), [
    ({"combined_video_judge_approved": True}, "create_subtitles"),
    ({"combined_video_judge_retry_target": "image_generation"}, "create_storyboard"),
    ({"combined_video_judge_retry_target": "motion_planner"}, "plan_motion"),
    ({"combined_video_judge_retry_target": "video_generation"}, "create_scene_videos"),
    ({"combined_video_judge_retry_target": "plan_edit_timeline"}, "plan_edit_timeline"),
    ({"combined_video_judge_error_exhausted": True}, END),
])
def test_combined_judge_routes(state_value, expected):
    assert agent_graph.route_combined_video_judge(state_value) == expected


@pytest.mark.parametrize(("source", "target", "retry_field"), [
    ("source_image", "image_generation", "shot_image_qa_retry_shots"),
    ("motion_plan", "motion_planner", "motion_plan_retry_shots"),
    ("video_generation", "video_generation", "video_retry_shots"),
    ("edit_timeline", "plan_edit_timeline", "combined_video_judge_retry_shots"),
])
def test_only_timestamp_affected_shot_is_retried(tmp_path, monkeypatch, source, target, retry_field):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_QA_MODEL", "ollama-vision-test")
    mock_frames(tmp_path, monkeypatch)
    invocation = {}

    def invoke(agent_name, prompt, image_files, *, purpose):
        invocation.update(agent_name=agent_name, prompt=prompt, image_files=image_files, purpose=purpose)
        return SimpleNamespace(content=json.dumps(
            report(source=source, target=target, issue=True, approved=False)
        )), {"agent": "combined-video-judge"}

    monkeypatch.setattr(judge, "invoke_with_images_and_evaluation", invoke)
    current = state(tmp_path)
    result = judge.combined_video_judge(current)
    assert result[retry_field] == ["shot-002"]
    assert result["combined_video_judge_retry_shots"] == ["shot-002"]
    assert invocation["agent_name"] == "combined-video-judge"
    assert invocation["image_files"][0].endswith("frame.jpg")
    assert result["combined_video_judge_model"] == "ollama-vision-test"


def test_missing_rough_cut_fails_safely(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    current = state(tmp_path)
    current["rough_cut_file"] = str(tmp_path / "missing.mp4")
    result = judge.combined_video_judge(current)
    assert result["combined_video_judge_status"] == "error"
    assert "missing or empty" in result["combined_video_judge_error"]


def test_ollama_failure_is_controlled(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    mock_frames(tmp_path, monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("Ollama request failed")

    monkeypatch.setattr(judge, "invoke_with_images_and_evaluation", fail)
    result = judge.combined_video_judge(state(tmp_path))
    assert result["combined_video_judge_status"] == "error"
    assert result["combined_video_judge_error_exhausted"] is False
    assert result["pipeline_status"] == "combined_video_judge_api_retry"


def test_malformed_ollama_response_fails_safely(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    mock_frames(tmp_path, monkeypatch)
    monkeypatch.setattr(judge, "invoke_with_images_and_evaluation", lambda *_args, **_kwargs: (
        SimpleNamespace(content='{"approved": true}'), {},
    ))
    result = judge.combined_video_judge(state(tmp_path))
    assert result["combined_video_judge_status"] == "error"
    assert "invalid structured output" in result["combined_video_judge_error"]


def test_refinement_retry_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    mock_frames(tmp_path, monkeypatch)
    monkeypatch.setattr(judge, "invoke_with_images_and_evaluation", lambda *_args, **_kwargs: (
        SimpleNamespace(content=json.dumps(
            report(source="video_generation", target="video_generation", issue=True, approved=False)
        )), {},
    ))
    current = state(tmp_path)
    current["video_judge_refinement_round"] = judge.COMBINED_VIDEO_JUDGE_CONFIG["max_refinement_rounds"]
    result = judge.combined_video_judge(current)
    assert result["video_judge_refinement_exhausted"] is True
    assert result["video_retry_shots"] == []
    assert agent_graph.route_combined_video_judge(result) == END


def test_ffmpeg_extracts_chronological_frames(tmp_path):
    video = tmp_path / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
    ], check=True)
    frames, frame_times = judge.extract_video_frames(str(video), 1.0, str(tmp_path / "frames"))
    assert frames
    assert frame_times == sorted(frame_times)
