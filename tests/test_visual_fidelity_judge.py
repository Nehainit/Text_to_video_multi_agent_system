from pathlib import Path

import pytest

from video_automation.agents import visual_fidelity_judge as judge


def report(score=8):
    scores = {metric: score for metric in judge.METRICS}
    issues = [] if score >= 8 else [{
        "metric": metric,
        "severity": "major",
        "description": f"Visible weakness in {metric}.",
        "timestamp_or_range": "00:01-00:02",
    } for metric in judge.METRICS]
    return {"judge": "visual_fidelity", "scores": scores, "issues": issues}


def state(tmp_path: Path):
    files = {}
    for name in ("source.png", "character.png", "style.png", "video.mp4"):
        path = tmp_path / name
        path.write_bytes(b"test")
        files[name] = str(path)
    return {
        "shot_plan": [{"shot_id": "shot-001", "characters_present": ["character-001"]}],
        "image_prompt_requests": [{
            "shot_id": "shot-001",
            "character_reference_ids": ["character-001"],
            "location_reference": {"location_id": "location-001"},
        }],
        "generated_videos": [{"shot_id": "shot-001", "video_file": files["video.mp4"]}],
        "video_validation_results": [{"shot_id": "shot-001", "valid": True}],
        "image_files": [files["source.png"]],
        "production_bible": {"characters": [{"character_id": "character-001"}]},
        "character_reference_files": [files["character.png"]],
        "mood_board_file": files["style.png"],
        "llm_evaluations": [],
    }


def test_valid_report_schema_passes():
    assert judge._validate_report(report()) == report()


def test_low_score_requires_specific_evidence():
    value = report()
    value["scores"]["character_identity"] = 7
    with pytest.raises(RuntimeError, match="omitted evidence"):
        judge._validate_report(value)


def test_visual_fidelity_node_reviews_validated_shot(monkeypatch, tmp_path):
    captured = {}

    def fake_judge(video_file, image_files, payload):
        captured.update(video_file=video_file, image_files=image_files, payload=payload)
        return report(), {"agent": "visual-fidelity-judge"}

    monkeypatch.setattr(judge, "judge_visual_fidelity", fake_judge)
    result = judge.review_visual_fidelity(state(tmp_path))

    assert result["visual_fidelity_reports"] == {"shot-001": report()}
    assert captured["payload"]["reference_image_order"] == [
        "approved_source_image", "character_reference:character-001", "style_reference",
    ]
    assert len(captured["image_files"]) == 3


def test_visual_fidelity_node_rejects_unvalidated_video(tmp_path):
    invalid = state(tmp_path)
    invalid["video_validation_results"][0]["valid"] = False
    with pytest.raises(RuntimeError, match="only deterministically validated"):
        judge.review_visual_fidelity(invalid)
