import json

import motion_planner_agent


def test_motion_planner_ignores_model_ids_and_preserves_approved_duration(monkeypatch, tmp_path):
    image = tmp_path / "shot-001.png"
    image.write_bytes(b"image")
    valid = {
        "shot_id": "shot-001",
        "duration_seconds": 4.0,
        "subject_motion": {"description": "Leo breathes gently.", "intensity": "low"},
        "camera_motion": {
            "type": "subtle_push_in",
            "description": "Use a very slow push-in while preserving the medium framing.",
            "intensity": "very_low",
        },
        "environment_motion": {"description": "Leaves move gently.", "intensity": "very_low"},
        "preserve_identity": True,
        "preserve_composition": True,
        "video_prompt": "Animate gentle breathing and leaf movement with a subtle push-in while preserving identity and composition.",
        "negative_prompt": "No face distortion, extra limbs, new characters, text, logo, or watermark.",
        "needs_revision": False,
        "revision_reason": None,
    }

    class Model:
        model = "motion-test"

        def __init__(self):
            self.responses = iter([{**valid, "shot_id": "wrong", "duration_seconds": 8}])
            self.messages = []

        def invoke(self, messages):
            self.messages.append(messages)
            return type("Response", (), {
                "content": json.dumps(next(self.responses)),
                "usage_metadata": {"input_tokens": 20, "output_tokens": 10},
            })()

    model = Model()
    monkeypatch.setattr(motion_planner_agent, "load_model", lambda _name: model)
    result = motion_planner_agent.create_motion_plans({
        "shot_plan": [{
            "shot_id": "shot-001", "estimated_duration_seconds": 4,
            "source_segment_ids": ["segment-001"],
        }],
        "image_prompt_requests": [{
            "shot_id": "shot-001", "source_segment_ids": ["segment-001"],
            "character_reference_ids": ["leo_001"], "location_id": "location-001",
        }],
        "image_files": [str(image)],
        "shot_image_qa_results": [{
            "shot_id": "shot-001", "approved": True, "animation_ready": True,
            "issues": [], "retry_target": None,
        }],
        "narration_segment_timings": [{
            "segment_id": "segment-001", "start_sec": 0.0, "end_sec": 4.0,
        }],
    })

    assert result["motion_plans"] == [valid]
    assert result["motion_plan_needs_revision"] is False
    assert len(model.messages) == 1
    assert [item["purpose"] for item in result["llm_evaluations"]] == ["plan_shot_motion"]
