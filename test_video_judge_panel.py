import pytest

import video_judge_panel as panel


def scored_report(judge, metrics, score=8):
    return {"judge": judge, "scores": {metric: score for metric in metrics}, "issues": []}


def meta_report(*, approved=True, source="none", target="none"):
    return {
        "approved": approved,
        "overall_score": 8 if approved else 7,
        "final_scores": {metric: 8 for metric in panel.ALL_METRICS},
        "strong_metrics": list(panel.ALL_METRICS),
        "weak_metrics": [],
        "critical_issues": [],
        "failure_source": source,
        "retry_target": target,
        "refinement_needed": not approved,
        "refinement_brief": [] if approved else ["Correct the identified failure."],
    }


def state():
    return {
        "shot_plan": [{
            "shot_id": "shot-001", "source_segment_ids": ["segment-001"],
            "narration": "Raja walks home.",
        }],
        "motion_plans": [{
            "shot_id": "shot-001", "duration_seconds": 5,
            "video_prompt": "Raja walks slowly.",
        }],
        "generated_videos": [{
            "shot_id": "shot-001", "video_file": "shot.mp4",
            "requested_duration_seconds": 5,
        }],
        "video_validation_results": [{"shot_id": "shot-001", "valid": True}],
        "image_files": ["source.png"],
        "story_beats": [{"beat_id": "beat-001", "description": "Raja returns home."}],
        "narration_segments": [{
            "segment_id": "segment-001", "parent_beat_ids": ["beat-001"],
            "text": "Raja walks home.",
        }],
        "visual_fidelity_reports": {
            "shot-001": scored_report("visual_fidelity", panel.VISUAL_METRICS),
        },
        "motion_temporal_reports": {
            "shot-001": scored_report("motion_temporal", panel.MOTION_METRICS),
        },
        "context_intent_reports": {
            "shot-001": scored_report("context_intent", panel.CONTEXT_METRICS),
        },
        "adversarial_reports": {"shot-001": {"challenges": [], "missed_issues": []}},
        "llm_evaluations": [],
    }


def test_motion_judge_receives_plan_prompt_duration_and_source_image(monkeypatch):
    captured = {}

    def invoke(**kwargs):
        captured.update(kwargs)
        return scored_report("motion_temporal", panel.MOTION_METRICS), {}

    monkeypatch.setattr(panel, "invoke_gemini_judge", invoke)
    result = panel.review_motion_temporal(state())

    assert result["motion_temporal_reports"]["shot-001"]["judge"] == "motion_temporal"
    assert captured["payload"]["approved_video_prompt"] == "Raja walks slowly."
    assert captured["payload"]["expected_duration_seconds"] == 5
    assert captured["image_files"] == ["source.png"]


def test_context_judge_receives_traceable_story_and_narration(monkeypatch):
    captured = {}

    def invoke(**kwargs):
        captured.update(kwargs)
        return scored_report("context_intent", panel.CONTEXT_METRICS), {}

    monkeypatch.setattr(panel, "invoke_gemini_judge", invoke)
    panel.review_context_intent(state())

    assert captured["payload"]["source_story_beats"][0]["beat_id"] == "beat-001"
    assert captured["payload"]["narration_segments"][0]["segment_id"] == "segment-001"


def test_adversarial_challenge_must_match_and_lower_original_score():
    primary = {
        "visual_fidelity": scored_report("visual_fidelity", panel.VISUAL_METRICS),
        "motion_temporal": scored_report("motion_temporal", panel.MOTION_METRICS),
        "context_intent": scored_report("context_intent", panel.CONTEXT_METRICS),
    }
    value = {
        "challenges": [{
            "judge": "visual_fidelity", "metric": "character_identity",
            "original_score": 8, "recommended_score": 9,
            "reason": "Brief face drift.", "timestamp_or_range": "00:02",
        }],
        "missed_issues": [],
    }
    with pytest.raises(RuntimeError, match="lower the original"):
        panel.validate_adversarial_report(value, primary)


def test_meta_judge_receives_only_reports_and_routes_source_image(monkeypatch):
    captured = {}
    rejected = meta_report(approved=False, source="source_image", target="image_generation")
    rejected["final_scores"]["character_identity"] = 7
    rejected["strong_metrics"].remove("character_identity")
    rejected["weak_metrics"] = ["character_identity"]

    def invoke(**kwargs):
        captured.update(kwargs)
        return rejected, {}

    monkeypatch.setattr(panel, "invoke_gemini_judge", invoke)
    result = panel.run_meta_judge(state())

    assert set(captured["payload"]) == {
        "visual_fidelity", "motion_temporal", "context_intent", "adversarial",
    }
    assert "video_file" not in captured and "image_files" not in captured
    assert result["judge_decision"] == "REFINE"
    assert result["judge_retry_target"] == "image_generation"
    assert result["shot_image_qa_retry_shots"] == ["shot-001"]


def test_meta_approval_rejects_critical_score_below_eight():
    value = meta_report()
    value["final_scores"]["temporal_consistency"] = 7
    value["strong_metrics"].remove("temporal_consistency")
    value["weak_metrics"] = ["temporal_consistency"]
    with pytest.raises(RuntimeError, match="critical failure"):
        panel.validate_meta_report(value)


def test_meta_retry_exhaustion_stops_retry(monkeypatch):
    rejected = meta_report(approved=False, source="video_generation", target="video_generation")
    rejected["final_scores"]["visual_artifact_control"] = 7
    rejected["strong_metrics"].remove("visual_artifact_control")
    rejected["weak_metrics"] = ["visual_artifact_control"]

    monkeypatch.setattr(panel, "invoke_gemini_judge", lambda **_kwargs: (rejected, {}))
    current = state()
    current["judge_retry_counts"] = {"shot-001": panel.JUDGE_PANEL_CONFIG["max_retries"]}
    result = panel.run_meta_judge(current)

    assert result["judge_exhausted_shots"] == ["shot-001"]
    assert result["judge_retry_shots"] == []
    assert result["pipeline_status"] == "judge_retry_exhausted"


def test_meta_cannot_approve_panel_critical_issue(monkeypatch):
    current = state()
    current["visual_fidelity_reports"]["shot-001"]["issues"] = [{
        "metric": "character_identity", "severity": "critical",
        "description": "Character identity changes briefly.", "timestamp_or_range": "00:02",
    }]
    monkeypatch.setattr(panel, "invoke_gemini_judge", lambda **_kwargs: (meta_report(), {}))

    with pytest.raises(RuntimeError, match="reported by the panel"):
        panel.run_meta_judge(current)
