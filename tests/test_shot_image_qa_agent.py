import json
from types import SimpleNamespace

from PIL import Image

from video_automation.agents import shot_image_qa_agent


def test_shot_image_qa_visually_reviews_each_shot_and_targets_only_failure(monkeypatch, tmp_path):
    generated, reference, mood = (tmp_path / name for name in ("shot.png", "leo.png", "mood.png"))
    for path in (generated, reference, mood):
        Image.new("RGB", (8, 8), "gold").save(path)
    verdicts = iter([
        {
            "shot_id": "wrong", "approved": False, "animation_ready": False,
            "issues": [{
                "type": "character_identity_drift", "severity": "major",
                "description": "Leo's mane differs from leo_001.",
                "correction": "Preserve leo_001's approved golden mane.",
            }],
            "retry_target": None,
        },
    ])
    calls = []

    def review(agent, prompt, image_files, *, purpose):
        calls.append((agent, prompt, image_files, purpose))
        return SimpleNamespace(content=json.dumps(next(verdicts))), {
            "agent": agent, "purpose": purpose, "model": "vision-test",
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            "token_source": "provider", "cost_usd": 0.001,
        }

    monkeypatch.setattr(shot_image_qa_agent, "invoke_with_images_and_evaluation", review)
    result = shot_image_qa_agent.review_shot_images({
        "shot_plan": [{"shot_id": "shot-001", "characters_present": ["leo_001"]}],
        "image_prompt_requests": [{
            "shot_id": "shot-001", "character_reference_ids": ["leo_001"],
            "location_reference": {"location_id": "location-001"},
        }],
        "generated_images": [{
            "shot_id": "shot-001", "generation_status": "success", "image_path": str(generated),
        }],
        "production_bible": {"characters": [{"character_id": "leo_001"}]},
        "character_reference_files": [str(reference)],
        "mood_board_file": str(mood),
        "shot_image_qa_pending_shots": ["shot-001"],
    })

    assert calls[0][0] == "shot-image-qa"
    assert calls[0][2] == [str(generated), str(reference), str(mood)]
    assert result["shot_image_qa_retry_shots"] == ["shot-001"]
    assert result["shot_image_qa_results"][0]["shot_id"] == "shot-001"
    assert result["shot_image_qa_results"][0]["retry_target"] == "image_generation"
    assert result["shot_image_qa_results"][0]["issues"][0]["type"] == "character_identity_drift"
    assert result["llm_evaluations"][0]["cost_usd"] == 0.001
