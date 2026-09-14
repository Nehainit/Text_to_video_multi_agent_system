import edit_timeline_agent as planner


def _state(second_duration=3.0, retry_counts=None):
    return {
        "actual_narration_seconds": 5.0,
        "shot_plan": [
            {
                "shot_id": "shot-001", "scene_id": "scene-001",
                "source_segment_ids": ["segment-001"], "visual_beat_ids": ["beat-001"],
                "estimated_duration_seconds": 2.0, "shot_purpose": "Establish the setting.",
            },
            {
                "shot_id": "shot-002", "scene_id": "scene-001",
                "source_segment_ids": ["segment-002"], "visual_beat_ids": ["beat-002"],
                "estimated_duration_seconds": 3.0, "shot_purpose": "Show the action.",
            },
        ],
        "generated_videos": [
            {"shot_id": "shot-001", "video_file": "one.mp4"},
            {"shot_id": "shot-002", "video_file": "two.mp4"},
        ],
        "video_validation_results": [
            {"shot_id": "shot-001", "valid": True, "metadata": {"duration_seconds": 2.0}},
            {"shot_id": "shot-002", "valid": True, "metadata": {"duration_seconds": second_duration}},
        ],
        "edit_timeline_retry_counts": retry_counts or {},
    }


def test_builds_contiguous_narration_timeline():
    result = planner.plan_edit_timeline(_state())
    assert result["timeline_valid"] is True
    assert result["timeline_duration_seconds"] == 5.0
    assert [(item["timeline_start"], item["timeline_end"]) for item in result["edit_timeline"]] == [(0.0, 2.0), (2.0, 5.0)]
    assert result["edit_timeline"][1]["narration_segment_ids"] == ["segment-002"]
    assert result["edit_timeline"][1]["visual_beat_ids"] == ["beat-002"]


def test_only_short_clip_is_retried():
    result = planner.plan_edit_timeline(_state(second_duration=2.0))
    assert result["timeline_valid"] is False
    assert result["video_retry_shots"] == ["shot-002"]
    assert result["edit_timeline_retry_counts"] == {"shot-002": 1}
    assert result["timeline_retry_requests"][0]["retry_target"] == "video_generation"


def test_exhausted_clip_uses_terminal_state():
    result = planner.plan_edit_timeline(_state(second_duration=2.0, retry_counts={"shot-002": 2}))
    assert result["video_retry_shots"] == []
    assert result["edit_timeline_exhausted_shots"] == ["shot-002"]
    assert result["pipeline_status"] == "edit_timeline_failed"
