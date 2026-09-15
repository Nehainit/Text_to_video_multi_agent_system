import json

from video_automation.agents import preproduction_agents
from video_automation.continuity_context import character_definitions


NARRATION = "Raja leaves the village, crosses the forest, and finally returns home safely."


def test_continuity_uses_backend_assigned_story_characters():
    characters = [{"character_id": "wrong", "name": "Robot", "description": "A lost robot."}]

    assert character_definitions({"story_outline": {"characters": characters}}) == [{
        "character_id": "character-001", "name": "Robot", "description": "A lost robot."
    }]


def scene(narration, purpose):
    return {
        "narration": narration,
        "location": "village and forest",
        "characters_present": ["raja_001"],
        "action": narration,
        "emotion": "determined",
        "story_purpose": purpose,
    }


def beat(scene_id, narration, purpose):
    return {
        "scene_id": scene_id,
        "narration": narration,
        "characters_present": ["raja_001"],
        "visual_action": narration,
        "emotional_intent": "determined",
        "continuity_in": "Raja moves right in a red coat.",
        "continuity_out": "Raja remains moving right in a red coat.",
        "story_purpose": purpose,
    }


def shot(scene_id, narration, purpose):
    return {
        **beat(scene_id, narration, purpose),
        "duration_seconds": 5,
        "visuals": narration,
        "camera": "wide 35mm eye level tracking composition",
        "motion": "pan_right",
        "subject_motion": "Raja walks naturally toward frame right.",
        "transition_to_next": {"type": "cut", "duration_seconds": 0, "audio_bridge": "none"},
        "sfx_prompt": "Soft footsteps and wind.",
    }


def planned_shot(scene_id, visual_beat_id, source_segment_ids, location_id, duration, action, purpose):
    return {
        "shot_id": "shot-001" if visual_beat_id == "vb-001" else "shot-002",
        "scene_id": scene_id,
        "visual_beat_ids": [visual_beat_id],
        "source_segment_ids": source_segment_ids,
        "characters_present": ["raja_001"],
        "primary_subject": "raja_001",
        "visual_action": action,
        "visual_focus": action.rstrip("."),
        "framing": "medium",
        "camera_angle": "eye_level",
        "composition": (
            "Raja stands in the foreground on the forest path, with the path extending behind him between the trees."
            if visual_beat_id == "vb-001"
            else "Raja occupies the foreground while the entrance of his home remains clearly visible behind him."
        ),
        "emotion": "determined" if visual_beat_id == "vb-001" else "relieved",
        "location_id": location_id,
        "estimated_duration_seconds": duration,
        "continuity_in": (
            "Raja has just left the village and is beginning his journey along the forest path."
            if visual_beat_id == "vb-001" else "Raja has reached the path outside his home."
        ),
        "continuity_out": (
            "Raja continues deeper along the same forest path toward home."
            if visual_beat_id == "vb-001" else "Raja remains safely outside his home, visibly relieved."
        ),
        "shot_purpose": purpose,
    }


class FakeModel:
    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else "\n".join(message["content"] for message in prompt)
        if "Narration Script Generator" in text:
            value = {
                "narration_segments": [
                    {"segment_id": "segment-001", "parent_beat_ids": ["beat-001"], "text": "Raja leaves the village,"},
                    {"segment_id": "segment-002", "parent_beat_ids": ["beat-002"], "text": "crosses the forest,"},
                    {"segment_id": "segment-003", "parent_beat_ids": ["beat-003"], "text": "and finally returns home safely."},
                ],
                "needs_story_revision": False,
                "revision_reason": None,
            }
        elif "Scene Planning Agent" in text:
            value = {
                "commentary": "Generated from narration timing.",
                "needs_revision": False,
                "revision_reason": None,
                "scenes": [
                    {
                        "scene_id": "scene-001",
                        "source_segment_ids": ["segment-001", "segment-002"],
                        "start_sec": 0.0,
                        "end_sec": 5.0,
                        "location_id": "location-001",
                        "characters_present": ["raja_001"],
                        "scene_goal": "Raja begins his journey.",
                        "visible_actions": ["Raja leaves the village.", "Raja crosses the forest."],
                        "emotion": "determined",
                        "story_purpose": "opening",
                    },
                    {
                        "scene_id": "scene-002",
                        "source_segment_ids": ["segment-003"],
                        "start_sec": 5.0,
                        "end_sec": 8.0,
                        "location_id": "location-002",
                        "characters_present": ["raja_001"],
                        "scene_goal": "Raja completes the journey.",
                        "visible_actions": ["Raja returns home safely."],
                        "emotion": "relieved",
                        "story_purpose": "resolution",
                    },
                ],
            }
        elif "Visual Planner Agent" in text:
            value = {
                "commentary": "Generated from approved scenes.",
                "needs_revision": False,
                "revision_reason": None,
                "scenes": [
                    {"visual_beats": [{
                        "visual_action": "Raja leaves the village and crosses the forest.",
                        "visual_focus": "Raja on the forest path",
                        "emotional_intent": "determination",
                        "continuity_in": "Raja starts at the village edge.",
                        "continuity_out": "Raja reaches the far side of the forest.",
                        "story_purpose": "opening",
                    }]},
                    {"visual_beats": [{
                        "visual_action": "Raja arrives home safely.",
                        "visual_focus": "Raja at home",
                        "emotional_intent": "relief",
                        "continuity_in": "Raja approaches his home.",
                        "continuity_out": "Raja stands safely at home.",
                        "story_purpose": "resolution",
                    }]},
                ],
            }
        elif "Shot Planner Agent" in text:
            value = {
                "commentary": "Generated from approved visual beats.",
                "needs_revision": False,
                "revision_reason": None,
                "shots": [
                    planned_shot(
                        "scene-001", "vb-001", ["segment-001", "segment-002"], "location-001", 5,
                        "Raja follows the forest path away from the village.", "Show Raja's journey.",
                    ),
                    planned_shot(
                        "scene-002", "vb-002", ["segment-003"], "location-002", 3,
                        "Raja stands safely outside his home.", "Show the resolved homecoming.",
                    ),
                ],
            }
        else:
            value = {"approved": True, "issues": []}
        return type("Response", (), {"content": json.dumps(value)})()


def test_preproduction_agents_keep_narration_and_visual_plan_aligned(monkeypatch):
    monkeypatch.setitem(
        preproduction_agents.SCENE_PLANNING_CONFIG["scene_boundaries"],
        "force_one_segment_per_scene",
        False,
    )
    monkeypatch.setitem(
        preproduction_agents.VISUAL_PLANNING_CONFIG["coverage"],
        "require_all_required_scene_actions_represented",
        False,
    )
    original = preproduction_agents.load_model
    preproduction_agents.load_model = lambda _name: FakeModel()
    state = {
        "topic": "Raja returns home",
        "story": NARRATION,
        "story_outline": {
            "idea": "Raja completes a difficult homeward journey.",
            "structure": [
                {"beat_id": "beat-001", "description": "Raja leaves the village."},
                {"beat_id": "beat-002", "description": "He crosses the forest."},
                {"beat_id": "beat-003", "description": "He returns home safely."},
            ],
        },
        "parsed_requirements": {"duration_seconds": 10, "language": "English"},
        "story_beats": [],
        "duration": "10 seconds",
        "language": "English",
        "production_bible": {
            "characters": [{"character_id": "raja_001", "name": "Raja"}],
            "locations": [
                {"location_id": "location-001", "description": "Village and forest path"},
                {"location_id": "location-002", "description": "Raja's home"},
            ],
        },
    }
    try:
        state.update(preproduction_agents.create_narration_script(state))
        state.update({
            "actual_narration_seconds": 8.0,
            "narration_segment_timings": [
                {"segment_id": "segment-001", "start_seconds": 0.0, "end_seconds": 3.0},
                {"segment_id": "segment-002", "start_seconds": 3.0, "end_seconds": 5.0},
                {"segment_id": "segment-003", "start_seconds": 5.0, "end_seconds": 8.0},
            ],
        })
        for creator in (
            preproduction_agents.plan_scenes,
            preproduction_agents.create_visual_beats,
            preproduction_agents.plan_shots,
            preproduction_agents.critique_shots,
        ):
            state.update(creator(state))
    finally:
        preproduction_agents.load_model = original

    assert len(state["scene_analysis"]) == 2
    assert state["scenes"][0]["source_segment_ids"] == ["segment-001", "segment-002"]
    assert len(state["visual_beats"]) == len(state["storyboard"]) == 2
    assert " ".join(item["narration"] for item in state["storyboard"]) == NARRATION
    assert sum(item["duration_seconds"] for item in state["storyboard"]) == 8
    assert state["shot_plan"][0]["visual_beat_ids"] == ["vb-001"]
    assert any(item["purpose"] == "generate_shot_plan" for item in state["llm_evaluations"])
    assert state["llm_evaluations"][0]["agent"] == "narration"
    assert [segment["parent_beat_ids"] for segment in state["narration_segments"]] == [
        ["beat-001"], ["beat-002"], ["beat-003"]
    ]
    assert state["estimated_narration_seconds"] == 8.0


def test_director_rejection_is_advisory_and_keeps_current_plan(monkeypatch):
    class DirectorModel:
        def invoke(self, _prompt):
            return type("Response", (), {"content": json.dumps({
                "approved": False,
                "issues": ["shot-001 framing could be stronger."],
            })})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: DirectorModel())
    result = preproduction_agents.critique_shots({
        "narration_script": "Raja returns home.",
        "visual_beats": [{"visual_beat_id": "vb-001"}],
        "production_bible": {},
        "shot_plan": [{"shot_id": "shot-001"}],
        "director_plan": [{"shot_id": "shot-001"}],
        "storyboard": [{"shot_id": "shot-001"}],
        "warnings": [],
    })

    assert result["director_approved"] is True
    assert result["storyboard"] == [{"shot_id": "shot-001"}]
    assert result["critic_issues"] == ["shot-001 framing could be stronger."]
    assert result["warnings"] == ["Director review warning: shot-001 framing could be stronger."]


def test_scene_review_rejection_keeps_last_structurally_valid_plan(monkeypatch):
    scene_plan = {
        "needs_revision": False,
        "revision_reason": None,
        "scenes": [{"scene_id": "scene-001", "start_sec": 0.0, "end_sec": 2.0}],
    }

    class RejectingModel:
        def invoke(self, prompt):
            content = prompt[0]["content"]
            value = (
                {"approved": False, "issues": ["The scene could be more faithful."]}
                if content == preproduction_agents.SCENE_FAITHFULNESS_REVIEW_SYSTEM_PROMPT
                else scene_plan
            )
            return type("Response", (), {"content": json.dumps(value)})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: RejectingModel())
    monkeypatch.setattr(
        preproduction_agents,
        "_validate_scene_plan",
        lambda _data, _state: (scene_plan["scenes"], [{"scene_id": "scene-001"}]),
    )
    monkeypatch.setattr(preproduction_agents, "_complete_scene_plan", lambda data, _state: data)
    result = preproduction_agents.plan_scenes({"production_bible": {}, "warnings": []})

    assert result["scene_plan_needs_revision"] is False
    assert result["scenes"] == scene_plan["scenes"]
    assert result["warnings"] == ["Scene review warning: The scene could be more faithful."]


def test_visual_review_rejection_fails_closed_after_retries(monkeypatch):
    visual_plan = {
        "needs_revision": False,
        "revision_reason": None,
        "visual_beats": [{"visual_beat_id": "vb-001"}],
    }

    class RejectingModel:
        def invoke(self, prompt):
            content = prompt[0]["content"]
            value = (
                {"approved": False, "issues": ["The visual could be more faithful."]}
                if content == preproduction_agents.VISUAL_FAITHFULNESS_REVIEW_SYSTEM_PROMPT
                else visual_plan
            )
            return type("Response", (), {"content": json.dumps(value)})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: RejectingModel())
    monkeypatch.setattr(preproduction_agents, "_complete_visual_plan", lambda data, _state: data)
    monkeypatch.setattr(preproduction_agents, "_validate_visual_plan", lambda _data, _state: visual_plan["visual_beats"])
    result = preproduction_agents.create_visual_beats({"scenes": [{"scene_id": "scene-001"}], "warnings": []})

    assert result["visual_plan_needs_revision"] is True
    assert result["visual_beats"] == []
    assert result["visual_plan_revision_reason"] == "The visual could be more faithful."
    assert result["planning_attempts"]["visual"]["used"] == 3


def test_narration_requests_story_revision_instead_of_changing_plot(monkeypatch):
    class RevisionModel:
        def invoke(self, prompt):
            assert prompt[0]["role"] == "system"
            assert prompt[0]["content"] == preproduction_agents.NARRATION_SYSTEM_PROMPT
            return type("Response", (), {"content": json.dumps({
                "narration_segments": [],
                "needs_story_revision": True,
                "revision_reason": "beat-002 contradicts beat-003, so their event order cannot be narrated coherently.",
            })})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: RevisionModel())
    result = preproduction_agents.create_narration_script({
        "duration": "10 seconds",
        "language": "English",
        "story_outline": {
            "idea": "A contradictory journey.",
            "structure": [
                {"beat_id": "beat-001", "description": "Raja leaves."},
                {"beat_id": "beat-002", "description": "Raja is already home."},
                {"beat_id": "beat-003", "description": "Raja has never left."},
            ],
        },
        "parsed_requirements": {"duration_seconds": 10, "language": "English"},
    })

    assert result["needs_story_revision"] is True
    assert result["narration_segments"] == []
    assert result["review_note"] == result["story_revision_reason"]


def test_narration_regenerates_when_text_is_not_supported_by_parent_beat(monkeypatch):
    unsupported = {
        "narration_segments": [
            {"segment_id": "segment-001", "parent_beat_ids": ["beat-001"], "text": "Raja finds a ripe mango."},
            {"segment_id": "segment-002", "parent_beat_ids": ["beat-002"], "text": "He shares half with Chuha."},
            {"segment_id": "segment-003", "parent_beat_ids": ["beat-003"], "text": "A golden bell rings while they celebrate."},
        ],
        "needs_story_revision": False,
        "revision_reason": None,
    }
    corrected = {
        **unsupported,
        "narration_segments": [
            *unsupported["narration_segments"][:2],
            {"segment_id": "segment-003", "parent_beat_ids": ["beat-003"], "text": "Together, they eat it happily."},
        ],
    }
    issue = 'segment-003: "golden bell rings" is unsupported by beat-003.'

    class TraceabilityModel:
        def __init__(self):
            self.responses = iter([
                unsupported,
                {"approved": False, "issues": [issue]},
                corrected,
                {"approved": True, "issues": []},
            ])
            self.prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return type("Response", (), {"content": json.dumps(next(self.responses))})()

    model = TraceabilityModel()
    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: model)
    result = preproduction_agents.create_narration_script({
        "duration": "10 seconds",
        "language": "English",
        "story_outline": {
            "idea": "Raja and Chuha share a mango.",
            "structure": [
                {"beat_id": "beat-001", "description": "Raja finds a ripe mango."},
                {"beat_id": "beat-002", "description": "Raja gives half of the mango to Chuha."},
                {"beat_id": "beat-003", "description": "Raja and Chuha eat the mango happily."},
            ],
        },
        "parsed_requirements": {"duration_seconds": 12, "language": "English"},
    })

    assert "golden bell" not in result["narration_script"]
    assert issue in model.prompts[2][1]["content"]
    assert [item["purpose"] for item in result["llm_evaluations"]] == [
        "generate_narration_script",
        "review_narration_traceability",
        "generate_narration_script",
        "review_narration_traceability",
    ]


def test_scene_planner_normalizes_timing_then_checks_faithfulness(monkeypatch):
    monkeypatch.setitem(
        preproduction_agents.SCENE_PLANNING_CONFIG["scene_boundaries"],
        "force_one_segment_per_scene",
        False,
    )

    def candidate(*, end=4.0, action="Raja and Chuha eat the mango together."):
        return {
            "needs_revision": False,
            "revision_reason": None,
            "scenes": [{
                "scene_id": "scene-001",
                "source_segment_ids": ["segment-001", "segment-002"],
                "start_sec": 0.0,
                "end_sec": end,
                "location_id": "location-001",
                "characters_present": ["raja_001", "chuha_001"],
                "scene_goal": "The friends share their mango.",
                "visible_actions": [action],
                "emotion": "happy cooperation",
                "story_purpose": "Resolve the story through sharing.",
            }],
        }

    issue = "scene-001 invents a golden bell that is unsupported by the narration."

    class SceneModel:
        def __init__(self):
            self.responses = iter([
                candidate(end=3.0),
                {"approved": False, "issues": [issue]},
                candidate(),
                {"approved": True, "issues": []},
            ])
            self.prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return type("Response", (), {"content": json.dumps(next(self.responses))})()

    model = SceneModel()
    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: model)
    result = preproduction_agents.plan_scenes({
        "story_outline": {"idea": "Two friends share fruit.", "structure": []},
        "parsed_requirements": {"topic": "Sharing", "duration_seconds": 4},
        "narration_segments": [
            {"segment_id": "segment-001", "parent_beat_ids": ["beat-001"], "text": "Raja finds a mango."},
            {"segment_id": "segment-002", "parent_beat_ids": ["beat-002"], "text": "He shares it with Chuha."},
        ],
        "narration_segment_timings": [
            {"segment_id": "segment-001", "start_seconds": 0.0, "end_seconds": 2.0},
            {"segment_id": "segment-002", "start_seconds": 2.0, "end_seconds": 4.0},
        ],
        "actual_narration_seconds": 4.0,
        "production_bible": {
            "characters": [
                {"character_id": "raja_001", "name": "Raja"},
                {"character_id": "chuha_001", "name": "Chuha"},
            ],
            "locations": [{"location_id": "location-001", "description": "Mango grove"}],
        },
    })

    assert result["scene_plan_needs_revision"] is False
    assert result["scenes"][0]["source_segment_ids"] == ["segment-001", "segment-002"]
    assert result["scenes"][0]["visible_actions"] == ["Raja and Chuha eat the mango together."]
    assert issue in model.prompts[2][1]["content"]
    assert [item["purpose"] for item in result["llm_evaluations"]] == [
        "generate_scene_plan", "review_scene_plan_faithfulness",
        "generate_scene_plan", "review_scene_plan_faithfulness",
    ]


def test_scene_planner_derives_ordered_segments_and_timing(monkeypatch):
    monkeypatch.setitem(
        preproduction_agents.SCENE_PLANNING_CONFIG["scene_boundaries"],
        "force_one_segment_per_scene",
        False,
    )

    class SceneModel:
        def invoke(self, prompt):
            if prompt[0]["content"] == preproduction_agents.SCENE_FAITHFULNESS_REVIEW_SYSTEM_PROMPT:
                return type("Response", (), {"content": json.dumps({"approved": True, "issues": []})})()
            assert json.loads(prompt[1]["content"])["user_requirements"]["topic"] == "A three-scene journey"
            return type("Response", (), {"content": json.dumps({
                "needs_revision": False,
                "revision_reason": None,
                "scenes": [
                    {
                        "segment_count": 2,
                        "location": "Forest path",
                        "characters_present": ["character-001"],
                        "scene_goal": "Raja travels.",
                        "visible_actions": ["Raja crosses the forest."],
                        "emotion": "determined",
                        "story_purpose": "Show the journey.",
                    },
                    {
                        "segment_count": 99,
                        "location": "Raja's home",
                        "characters_present": ["character-001"],
                        "scene_goal": "Raja arrives home.",
                        "visible_actions": ["Raja reaches home."],
                        "emotion": "relieved",
                        "story_purpose": "Resolve the journey.",
                    },
                ],
            })})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: SceneModel())
    result = preproduction_agents.plan_scenes({
        "narration_segments": [
            {"segment_id": f"segment-{index:03}", "parent_beat_ids": [], "text": str(index)}
            for index in range(1, 4)
        ],
        "narration_segment_timings": [
            {"segment_id": f"segment-{index:03}", "start_seconds": index - 1, "end_seconds": index}
            for index in range(1, 4)
        ],
        "actual_narration_seconds": 3.0,
        "parsed_requirements": {
            "topic": "A three-scene journey",
            "characters": ["Raja"],
            "visual_style": "watercolor",
        },
    })

    assert [scene["source_segment_ids"] for scene in result["scenes"]] == [
        ["segment-001", "segment-002"], ["segment-003"]
    ]
    assert [(scene["start_sec"], scene["end_sec"]) for scene in result["scenes"]] == [(0.0, 2.0), (2.0, 3.0)]
    assert [scene["location_id"] for scene in result["scenes"]] == ["location-001", "location-002"]


def test_scene_planner_rejects_collapsing_multiple_segments_into_one_scene(monkeypatch):
    class SceneModel:
        def invoke(self, _prompt):
            return type("Response", (), {"content": json.dumps({
                "needs_revision": False,
                "revision_reason": None,
                "scenes": [{
                    "segment_count": 3,
                    "location": "Enchanted forest",
                    "characters_present": [],
                    "scene_goal": "Complete the journey.",
                    "visible_actions": ["The travellers cross the forest and reach the palace."],
                    "emotion": "hopeful",
                    "story_purpose": "Show the complete journey.",
                }],
            })})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: SceneModel())
    result = preproduction_agents.plan_scenes({
        "narration_segments": [
            {"segment_id": f"segment-{index:03}", "parent_beat_ids": [], "text": str(index)}
            for index in range(1, 4)
        ],
        "narration_segment_timings": [
            {"segment_id": f"segment-{index:03}", "start_seconds": index - 1, "end_seconds": index}
            for index in range(1, 4)
        ],
        "actual_narration_seconds": 3.0,
        "parsed_requirements": {"characters": []},
    })

    assert result["scene_plan_needs_revision"] is True
    assert result["scene_plan_revision_reason"] == "Every scene must map to exactly one narration segment."


def test_scene_planner_splits_one_segment_across_multiple_locations(monkeypatch):
    class SceneModel:
        def invoke(self, prompt):
            if prompt[0]["content"] == preproduction_agents.SCENE_FAITHFULNESS_REVIEW_SYSTEM_PROMPT:
                value = {"approved": True, "issues": []}
            else:
                value = {
                    "needs_revision": False,
                    "revision_reason": None,
                    "scenes": [
                        {
                            "source_segment_number": 1,
                            "location": location,
                            "characters_present": ["raja_001", "rani_001"],
                            "scene_goal": goal,
                            "visible_actions": [action],
                            "emotion": "hopeful",
                            "story_purpose": "Advance the journey home.",
                        }
                        for location, goal, action in [
                            ("Fairy glade", "Receive magical help.", "Raja and Rani meet a helpful fairy."),
                            ("River of fire", "Cross the dangerous river.", "Raja and Rani cross the river of fire."),
                            ("Maze of illusions", "Find the path through the maze.", "Raja and Rani navigate the maze."),
                        ]
                    ],
                }
            return type("Response", (), {"content": json.dumps(value)})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: SceneModel())
    result = preproduction_agents.plan_scenes({
        "narration_segments": [{
            "segment_id": "segment-001",
            "parent_beat_ids": ["beat-001"],
            "text": "A fairy helps Raja and Rani cross a river of fire and navigate a maze.",
        }],
        "narration_segment_timings": [{
            "segment_id": "segment-001", "start_seconds": 0.0, "end_seconds": 9.0,
        }],
        "actual_narration_seconds": 9.0,
        "production_bible": {"characters": [
            {"character_id": "raja_001", "name": "Raja"},
            {"character_id": "rani_001", "name": "Rani"},
        ]},
    })

    assert [scene["source_segment_ids"] for scene in result["scenes"]] == [["segment-001"]] * 3
    assert [(scene["start_sec"], scene["end_sec"]) for scene in result["scenes"]] == [
        (0.0, 3.0), (3.0, 6.0), (6.0, 9.0),
    ]
    assert [scene["location_id"] for scene in result["scenes"]] == [
        "location-001", "location-002", "location-003",
    ]


def test_scene_planner_accepts_populated_scenes_when_model_revision_flag_disagrees(monkeypatch):
    class SceneModel:
        format = "json"

        def __init__(self):
            self.formats = []

        def invoke(self, prompt):
            self.formats.append(self.format)
            if prompt[0]["content"] == preproduction_agents.SCENE_FAITHFULNESS_REVIEW_SYSTEM_PROMPT:
                return type("Response", (), {"content": json.dumps({"approved": True, "issues": []})})()
            return type("Response", (), {"content": json.dumps({
                "needs_revision": True,
                "revision_reason": "Every scene needs a nonempty emotion.",
                "scenes": [{
                    "segment_count": 1,
                    "location": "Park",
                    "characters_present": [],
                    "scene_goal": "Show the lost robot.",
                    "visible_actions": ["A robot looks around the park."],
                    "emotion": "confused",
                    "story_purpose": "Establish that the robot is lost.",
                }],
            })})()

    model = SceneModel()
    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: model)
    result = preproduction_agents.plan_scenes({
        "narration_segments": [{"segment_id": "segment-001", "parent_beat_ids": [], "text": "A robot is lost."}],
        "narration_segment_timings": [{"segment_id": "segment-001", "start_seconds": 0.0, "end_seconds": 1.0}],
        "actual_narration_seconds": 1.0,
        "parsed_requirements": {"characters": []},
    })

    assert result["scene_plan_needs_revision"] is False
    assert result["scenes"][0]["emotion"] == "confused"
    assert model.formats == [preproduction_agents.SCENE_PLAN_OUTPUT_SCHEMA, "json"]


def test_narration_leaves_estimated_length_to_tts_timing(monkeypatch):
    invalid = {
        "narration_segments": [
            {"segment_id": "segment-001", "parent_beat_ids": ["beat-001"], "text": "Too short."},
        ],
        "needs_story_revision": False,
        "revision_reason": None,
    }
    class NarrationModel:
        def __init__(self):
            self.responses = iter([invalid, {"approved": True, "issues": []}])

        def invoke(self, _prompt):
            return type("Response", (), {"content": json.dumps(next(self.responses))})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: NarrationModel())
    result = preproduction_agents.create_narration_script({
        "duration": "10 seconds",
        "language": "English",
        "story_outline": {"idea": "Raja returns.", "structure": [{"beat_id": "beat-001", "description": "Raja returns home."}]},
        "parsed_requirements": {"duration_seconds": 10, "language": "English"},
    })

    assert result["narration_script"] == "Too short."
    assert "TTS timing will decide" in result["warnings"][0]
    assert [item["purpose"] for item in result["llm_evaluations"]] == [
        "generate_narration_script", "review_narration_traceability"
    ]


def test_narration_retries_segment_count_and_derives_traceability(monkeypatch):
    wrong_count = {
        "narration_segments": [{"text": "Robby loses his route, searches a warehouse, and eventually finds his way home."}],
        "needs_story_revision": False,
        "revision_reason": None,
    }
    corrected = {
        **wrong_count,
        "narration_segments": [
            {"segment_id": "wrong", "parent_beat_ids": ["wrong"], "text": "Robby loses his delivery route."},
            {"segment_id": "wrong", "parent_beat_ids": [], "text": "He searches an abandoned warehouse."},
            {"segment_id": "wrong", "parent_beat_ids": ["beat-001"], "text": "A friend guides him home."},
        ],
    }

    class NarrationModel:
        def __init__(self):
            self.responses = iter([wrong_count, corrected, {"approved": True, "issues": []}])

        def invoke(self, _prompt):
            return type("Response", (), {"content": json.dumps(next(self.responses))})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: NarrationModel())
    result = preproduction_agents.create_narration_script({
        "duration": "10 seconds",
        "language": "English",
        "story_outline": {
            "idea": "Robby returns home.",
            "structure": [
                {"beat_id": "beat-001", "description": "Robby loses his delivery route."},
                {"beat_id": "beat-002", "description": "Robby searches an abandoned warehouse."},
                {"beat_id": "beat-003", "description": "A friend guides Robby home."},
            ],
        },
        "parsed_requirements": {"duration_seconds": 10, "language": "English"},
    })

    assert [segment["segment_id"] for segment in result["narration_segments"]] == [
        "segment-001", "segment-002", "segment-003"
    ]
    assert [segment["parent_beat_ids"] for segment in result["narration_segments"]] == [
        ["beat-001"], ["beat-002"], ["beat-003"]
    ]
    assert [item["purpose"] for item in result["llm_evaluations"]] == [
        "generate_narration_script", "generate_narration_script", "review_narration_traceability"
    ]


def test_narration_falls_back_to_approved_beats_after_malformed_json(monkeypatch):
    class BrokenNarrationModel:
        format = "json"

        def __init__(self):
            self.formats = []

        def invoke(self, _prompt):
            self.formats.append(self.format)
            return type("Response", (), {"content": '{"narration_segments":[{"text":"unfinished}'})()

    model = BrokenNarrationModel()
    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: model)
    result = preproduction_agents.create_narration_script({
        "duration": "10 seconds",
        "language": "English",
        "story_outline": {
            "structure": [
                {"beat_id": "beat-001", "description": "Ganesha enters the magical forest."},
                {"beat_id": "beat-002", "description": "He guides the mouse safely home."},
            ],
        },
        "parsed_requirements": {"duration_seconds": 10, "language": "English"},
    })

    assert result["narration_script"] == (
        "Ganesha enters the magical forest. He guides the mouse safely home."
    )
    assert [segment["parent_beat_ids"] for segment in result["narration_segments"]] == [
        ["beat-001"], ["beat-002"],
    ]
    assert all(value == preproduction_agents.NARRATION_OUTPUT_SCHEMA for value in model.formats)
    assert "using approved story beats" in result["warnings"][-1]


def test_scene_planner_surfaces_revision_instead_of_inventing_visuals(monkeypatch):
    class RevisionModel:
        def invoke(self, prompt):
            assert prompt[0]["content"] == preproduction_agents.SCENE_PLANNING_SYSTEM_PROMPT
            return type("Response", (), {"content": json.dumps({
                "needs_revision": True,
                "revision_reason": "segment-002 names no visible subject or action.",
                "scenes": [],
            })})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: RevisionModel())
    result = preproduction_agents.plan_scenes({
        "story": "An abstract thought occurs.",
        "narration_segments": [{"segment_id": "segment-001", "parent_beat_ids": ["beat-001"], "text": "Hope exists."}],
        "narration_segment_timings": [{"segment_id": "segment-001", "start_seconds": 0.0, "end_seconds": 1.0}],
        "actual_narration_seconds": 1.0,
        "production_bible": {"characters": [], "locations": [{"location_id": "location-001", "description": "Room"}]},
    })

    assert result["scene_plan_needs_revision"] is True
    assert result["scenes"] == []


def test_visual_planner_allows_multiple_beats_for_one_segment(monkeypatch):
    visual_plan = {
        "needs_revision": False,
        "revision_reason": None,
        "scenes": [{"visual_beats": [
            {
                "visual_action": "Raja notices the mango.", "visual_focus": "Raja and the mango",
                "emotional_intent": "curiosity", "continuity_in": "Raja stands below the tree.",
                "continuity_out": "Raja looks toward the branch.", "story_purpose": "Establish the goal.",
            },
            {
                "visual_action": "Raja reaches toward the mango.", "visual_focus": "Raja's hand and the mango",
                "emotional_intent": "effort", "continuity_in": "Raja looks toward the branch.",
                "continuity_out": "Raja holds the mango.", "story_purpose": "Complete the visible action.",
            },
        ]}],
    }
    scope_invalid = json.loads(json.dumps(visual_plan))
    scope_invalid["scenes"][0]["visual_beats"][0]["visual_action"] = "Camera zooms toward Raja."
    semantic_invalid = json.loads(json.dumps(visual_plan))
    semantic_invalid["scenes"][0]["visual_beats"][0]["visual_action"] = "Raja discovers a golden bell beside the mango."
    issue = "vb-001 invents a golden bell unsupported by its scene and source segment."

    class VisualModel:
        def __init__(self):
            self.responses = iter([
                scope_invalid,
                semantic_invalid,
                {"approved": False, "issues": [issue]},
                visual_plan,
                {"approved": True, "issues": []},
            ])
            self.prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return type("Response", (), {"content": json.dumps(next(self.responses))})()

    model = VisualModel()
    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: model)
    state = {
        "story_outline": {"idea": "Raja gets a mango.", "structure": [{"beat_id": "beat-001", "description": "Raja gets a mango."}]},
        "parsed_requirements": {"topic": "Raja gets a mango."},
        "narration_segments": [{"segment_id": "segment-001", "parent_beat_ids": ["beat-001"], "text": "Raja reaches for the mango."}],
        "narration_segment_timings": [{"segment_id": "segment-001", "start_seconds": 0.0, "end_seconds": 3.0}],
        "scenes": [{
            "scene_id": "scene-001", "source_segment_ids": ["segment-001"], "start_sec": 0.0, "end_sec": 3.0,
            "location_id": "location-001", "characters_present": ["raja_001"], "scene_goal": "Get the mango.",
            "visible_actions": ["Raja reaches for the mango."], "emotion": "effort", "story_purpose": "Resolve the goal.",
        }],
        "production_bible": {
            "characters": [{"character_id": "raja_001", "name": "Raja"}],
            "locations": [{"location_id": "location-001", "description": "Mango tree"}],
        },
    }

    result = preproduction_agents.create_visual_beats(state)
    assert [beat["visual_beat_id"] for beat in result["visual_beats"]] == ["vb-001", "vb-002"]
    assert [beat["scene_id"] for beat in result["visual_beats"]] == ["scene-001", "scene-001"]
    assert [beat["source_segment_ids"] for beat in result["visual_beats"]] == [["segment-001"], ["segment-001"]]
    assert [beat["parent_story_beat_ids"] for beat in result["visual_beats"]] == [["beat-001"], ["beat-001"]]
    assert result["count_adjustment"] == "Planned 2 visual beats dynamically across 1 scenes."
    assert "camera" in model.prompts[1][1]["content"].lower()
    assert issue in model.prompts[3][1]["content"]
    assert [item["purpose"] for item in result["llm_evaluations"]] == [
        "generate_visual_plan", "generate_visual_plan", "review_visual_plan_faithfulness",
        "generate_visual_plan", "review_visual_plan_faithfulness",
    ]


def test_visual_planner_splits_each_required_scene_action(monkeypatch):
    def visual_beat(action):
        return {
            "visual_action": action,
            "visual_focus": action.rstrip("."),
            "emotional_intent": "hopeful",
            "continuity_in": "Raja and Rani continue through the enchanted forest.",
            "continuity_out": action,
            "story_purpose": "Advance their journey home.",
        }

    actions = [
        "Raja and Rani meet a helpful fairy.",
        "Raja and Rani cross the river of fire.",
        "Raja and Rani navigate the maze of illusions.",
    ]
    collapsed = {
        "needs_revision": False,
        "revision_reason": None,
        "scenes": [{"visual_beats": [visual_beat(" ".join(actions))]}],
    }
    corrected = {
        "needs_revision": False,
        "revision_reason": None,
        "scenes": [{"visual_beats": [visual_beat(action) for action in actions]}],
    }

    class VisualModel:
        def __init__(self):
            self.responses = iter([collapsed, corrected, {"approved": True, "issues": []}])
            self.prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return type("Response", (), {"content": json.dumps(next(self.responses))})()

    model = VisualModel()
    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: model)
    result = preproduction_agents.create_visual_beats({
        "story_outline": {"structure": [{"beat_id": "beat-001"}]},
        "narration_segments": [{
            "segment_id": "segment-001",
            "parent_beat_ids": ["beat-001"],
            "text": "With magical help, Raja and Rani cross the river and maze.",
        }],
        "scenes": [{
            "scene_id": "scene-001",
            "source_segment_ids": ["segment-001"],
            "visible_actions": actions,
        }],
    })

    assert [beat["visual_action"] for beat in result["visual_beats"]] == actions
    assert "one focused visual beat" in model.prompts[1][1]["content"]


def test_visual_planner_accepts_populated_plan_when_revision_flag_disagrees(monkeypatch):
    class VisualModel:
        format = "json"

        def __init__(self):
            self.formats = []

        def invoke(self, prompt):
            self.formats.append(self.format)
            if prompt[0]["content"] == preproduction_agents.VISUAL_FAITHFULNESS_REVIEW_SYSTEM_PROMPT:
                return type("Response", (), {"content": json.dumps({"approved": True, "issues": []})})()
            return type("Response", (), {"content": json.dumps({
                "needs_revision": True,
                "revision_reason": "The prior visual response was invalid.",
                "scenes": [{"visual_beats": [{
                    "visual_action": "Robo looks around the city.",
                    "visual_focus": "Robo",
                    "emotional_intent": "curious",
                    "continuity_in": "Robo enters the city.",
                    "continuity_out": "Robo studies the street.",
                    "story_purpose": "Establish Robo's search.",
                }]}],
            })})()

    model = VisualModel()
    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: model)
    result = preproduction_agents.create_visual_beats({
        "story_outline": {"structure": [{"beat_id": "beat-001"}]},
        "narration_segments": [{"segment_id": "segment-001", "parent_beat_ids": ["beat-001"], "text": "Robo searches."}],
        "scenes": [{"scene_id": "scene-001", "source_segment_ids": ["segment-001"]}],
    })

    assert result["visual_plan_needs_revision"] is False
    assert result["visual_beats"][0]["visual_beat_id"] == "vb-001"
    assert model.formats[0]["properties"]["scenes"]["minItems"] == 1
    assert model.formats[0]["properties"]["scenes"]["maxItems"] == 1
    assert model.formats[1] == "json"


def test_shot_planner_surfaces_visual_revision(monkeypatch):
    class RevisionModel:
        def invoke(self, prompt):
            assert prompt[0]["content"] == preproduction_agents.SHOT_PLANNING_SYSTEM_PROMPT
            return type("Response", (), {"content": json.dumps({
                "needs_revision": True,
                "revision_reason": "vb-001 combines sequential actions that cannot form one clear still frame.",
                "shots": [],
            })})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: RevisionModel())
    result = preproduction_agents.plan_shots({"visual_beats": []})

    assert result["shot_plan_needs_revision"] is True
    assert result["shot_plan"] == []
    assert result["llm_evaluations"][0]["purpose"] == "generate_shot_plan"


def test_shot_planner_completes_deterministic_fields_and_stops_duplicate_failures(monkeypatch):
    state = {
        "scenes": [{
            "scene_id": "scene-001", "location_id": "location-001", "start_sec": 0.0, "end_sec": 8.0,
            "characters_present": ["raja_001"],
        }],
        "visual_beats": [
            {"visual_beat_id": "vb-001", "scene_id": "scene-001", "source_segment_ids": ["segment-001"]},
            {"visual_beat_id": "vb-002", "scene_id": "scene-001", "source_segment_ids": ["segment-002"]},
        ],
        "production_bible": {"characters": [{"character_id": "raja_001", "name": "Raja"}]},
    }
    raw = {
        "needs_revision": True,
        "revision_reason": "The prior shot response was invalid.",
        "shots": [
            {
                **{key: value for key, value in planned_shot("wrong", "vb-001", [], "wrong", 1, "Raja walks.", "Begin.").items()
                   if key in preproduction_agents.SHOT_CREATIVE_FIELDS},
                "visual_beat_count": 1,
            },
            {
                **{key: value for key, value in planned_shot("wrong", "vb-002", [], "wrong", 3, "Raja arrives.", "Finish.").items()
                   if key in preproduction_agents.SHOT_CREATIVE_FIELDS},
                "visual_beat_count": 1,
            },
        ],
    }

    completed = preproduction_agents._complete_shot_plan(raw, state)["shots"]

    assert [shot["shot_id"] for shot in completed] == ["shot-001", "shot-002"]
    assert [shot["source_segment_ids"] for shot in completed] == [["segment-001"], ["segment-002"]]
    assert [shot["estimated_duration_seconds"] for shot in completed] == [2.0, 6.0]

    calls = []

    class InvalidModel:
        def invoke(self, _prompt):
            calls.append(1)
            return type("Response", (), {"content": "{}"})()

    monkeypatch.setattr(preproduction_agents, "load_model", lambda _name: InvalidModel())
    failed = preproduction_agents.plan_shots(state)
    failed = preproduction_agents.plan_shots({
        **state,
        **failed,
        "shot_plan_feedback": failed["shot_plan_revision_reason"],
    })

    assert len(calls) == 2
    assert failed["planning_attempts"]["shot"]["used"] == 3


def test_shot_completion_derives_missing_continuity_from_visual_beats():
    state = {
        "scenes": [{
            "scene_id": "scene-001", "location_id": "location-001", "start_sec": 0.0, "end_sec": 4.0,
            "characters_present": ["raja_001"],
        }],
        "visual_beats": [{
            "visual_beat_id": "vb-001", "scene_id": "scene-001", "source_segment_ids": ["segment-001"],
            "continuity_in": "Raja waits at the forest entrance.",
            "continuity_out": "Raja steps into the forest.",
        }],
        "production_bible": {"characters": [{"character_id": "raja_001", "name": "Raja"}]},
    }
    raw = {
        "needs_revision": False, "revision_reason": None,
        "shots": [{
            "visual_beat_ids": ["vb-001"], "characters_present": ["raja_001"],
            "primary_subject": "raja_001", "visual_action": "Raja steps into the forest.",
            "visual_focus": "Raja at the forest entrance", "framing": "medium",
            "camera_angle": "eye_level", "composition": "Raja stands beside the entrance.",
            "emotion": "curious", "estimated_duration_seconds": 4,
            "shot_purpose": "Establish the journey.",
        }],
    }

    completed = preproduction_agents._complete_shot_plan(raw, state)["shots"][0]

    assert completed["continuity_in"] == "Raja waits at the forest entrance."
    assert completed["continuity_out"] == "Raja steps into the forest."
