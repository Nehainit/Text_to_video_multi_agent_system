from langgraph.types import Command

from video_automation import agent_graph


def shot(visuals="Opening resolves safely."):
    return {
        "shot_id": "shot-001",
        "shot_number": 1,
        "scene_id": "scene-001",
        "scene_number": 1,
        "story_purpose": "resolution",
        "duration_seconds": 10,
        "start_time": "00:00",
        "end_time": "00:10",
        "visuals": visuals,
        "camera": "wide 35mm eye level tracking composition",
        "motion": "zoom_in",
        "subject_motion": "Raja steps forward.",
        "continuity_in": "Raja in red coat at frame left.",
        "continuity_out": "Raja in red coat at frame center.",
        "transition_to_next": {"type": "cut", "duration_seconds": 0, "audio_bridge": "none"},
        "narration": "Raja returns home safely.",
        "sfx_prompt": "Soft wind.",
    }


def fake_story_creator(state):
    note = state.get("review_note", "")
    return {
        "story": f"Raja leaves, changes course, and returns home safely. {note}".strip(),
        "story_beats": [{"purpose": "opening"}, {"purpose": "turn"}, {"purpose": "resolution"}],
        "characters": ["Raja — red coat"],
        "review_note": "",
    }


def fake_reference_creator(state):
    suffix = "-revised" if state.get("reference_feedback") else ""
    return {
        "character_board_file": f"outputs/test/character{suffix}.png",
        "character_reference_files": [f"outputs/test/character{suffix}.png"],
        "mood_board_file": f"outputs/test/mood{suffix}.png",
        "reference_feedback": "",
        "last_reference_feedback": state.get("reference_feedback", ""),
        "reference_approved": False,
    }


def fake_narration_script_creator(state):
    return {"narration_script": state["story"]}


def fake_scene_planner(state):
    return {"scene_analysis": [{"scene_id": "scene-001", "narration": state["narration_script"]}]}


def fake_visual_beat_creator(state):
    return {"visual_beats": [{"beat_id": "beat-001", "scene_id": "scene-001", "narration": state["narration_script"]}]}


def fake_shot_planner(_state):
    plan = [shot()]
    return {
        "shot_plan": plan,
        "director_plan": plan,
        "storyboard": plan,
    }


def fake_director_critic(state):
    return {"director_plan": state["shot_plan"], "storyboard": state["shot_plan"], "director_approved": True}


def fake_image_prompt_builder(_state):
    return {"image_prompt_requests": [{"shot_id": "shot-001", "image_prompt": "Opening resolves safely."}]}


def fake_image_creator(state):
    feedback = state.get("visual_feedback", [])
    suffix = "-revised" if feedback else ""
    return {
        "image_files": [f"outputs/test/images/shot-001{suffix}.png"],
        "visual_feedback": [],
        "last_visual_feedback": feedback,
    }


def fake_shot_image_reviewer(_state):
    return {
        "shot_image_qa_results": [{
            "shot_id": "shot-001", "approved": True, "animation_ready": True,
            "issues": [], "retry_target": None,
        }],
        "shot_image_qa_retry_shots": [],
        "shot_image_qa_round": 1,
    }


def fake_motion_planner(_state):
    return {
        "motion_plans": [{
            "shot_id": "shot-001", "duration_seconds": 10,
            "video_prompt": "Raja takes one gentle step.", "negative_prompt": "No distortion.",
        }],
        "motion_plan_needs_revision": False,
    }


def fake_narration_creator(_state):
    return {
        "narration_file": "outputs/test/audio/narration.mp3",
        "scene_timings": [{"scene_number": 1, "start_seconds": 0, "end_seconds": 10}],
    }


def fake_scene_video_creator(_state):
    return {"scene_video_files": [None], "warnings": ["local fallback"]}


def fake_soundfx_creator(_state):
    return {"sfx_files": ["outputs/test/audio/sfx_shot_001.mp3"]}


def fake_subtitle_creator(state):
    return {"subtitles": [{"scene_number": 1, "text": state["story"]}], "subtitle_file": "outputs/test/subtitles/story.srt"}


def fake_video_creator(_state):
    return {"mixed_audio_file": "outputs/test/video/mixed_audio.mp3", "video_file": "outputs/test/video/final_reel.mp4"}


def fake_timeline_planner(_state):
    return {
        "timeline_valid": True,
        "video_retry_shots": [],
        "edit_timeline_exhausted_shots": [],
    }


def fake_rough_video_compiler(_state):
    return {
        "compilation_status": "success",
        "rough_cut_file": "outputs/test/rough_cut/rough_cut_v001.mp4",
        "sync_valid": True,
        "compilation_issues": [],
        "rough_cut_retry_exhausted": False,
    }


def fake_combined_video_judge(_state):
    return {
        "combined_video_judge_approved": True,
        "combined_video_judge_status": "accepted",
        "combined_video_judge_retry_target": "none",
        "video_judge_refinement_exhausted": False,
        "video_retry_shots": [],
        "pipeline_status": "combined_video_judge_accepted",
    }


def build_fake_graph(checkpointer=None, **overrides):
    creators = dict(
        story_creator=fake_story_creator,
        reference_creator=fake_reference_creator,
        narration_script_creator=fake_narration_script_creator,
        scene_planner=fake_scene_planner,
        visual_beat_creator=fake_visual_beat_creator,
        shot_planner=fake_shot_planner,
        director_critic=fake_director_critic,
        image_prompt_builder=fake_image_prompt_builder,
        image_creator=fake_image_creator,
        shot_image_reviewer=fake_shot_image_reviewer,
        motion_planner=fake_motion_planner,
        narration_creator=fake_narration_creator,
        scene_video_creator=fake_scene_video_creator,
        soundfx_creator=fake_soundfx_creator,
        subtitle_creator=fake_subtitle_creator,
        editor=fake_video_creator,
        video_validator=None,
        timeline_planner=fake_timeline_planner,
        rough_video_compiler=fake_rough_video_compiler,
        video_judge=fake_combined_video_judge,
        checkpointer=checkpointer,
    )
    creators.update(overrides)
    return agent_graph.build_story_graph(**creators)


def state():
    return {
        "topic": "Raja returns home",
        "tone": "mystical",
        "duration": "10 seconds",
        "language": "English",
        "characters": ["Raja"],
    }


def test_character_reference_is_automatic_then_visual_storyboard_is_reviewed(capsys):
    graph = build_fake_graph()
    config = {"configurable": {"thread_id": "automatic-reference"}}
    result = graph.invoke(state(), config)
    assert "story" in result["__interrupt__"][0].value
    output = capsys.readouterr().out
    assert "[pipeline:-] START create_story" in output
    assert "[pipeline:-] DONE create_story" in output
    assert "[pipeline:-] PAUSED review_story" in output

    result = graph.invoke(Command(resume={"action": "approve"}), config)
    assert "image_files" in result["__interrupt__"][0].value
    assert result["character_board_file"].endswith("character.png")

    result = graph.invoke(Command(resume={"action": "approve"}), config)
    assert result["video_file"].endswith("final_reel.mp4")
    assert result["warnings"] == ["local fallback"]


def test_story_feedback_is_applied_and_recorded():
    graph = build_fake_graph()
    config = {"configurable": {"thread_id": "story-feedback"}}
    graph.invoke(state(), config)

    revised = graph.invoke(Command(resume={"action": "regenerate", "note": "make the ending hopeful"}), config)
    review = revised["__interrupt__"][0].value
    assert "make the ending hopeful" in revised["story"]
    assert review["feedback"] == "make the ending hopeful"
    assert revised["story_feedback_history"] == [{"revision": 1, "note": "make the ending hopeful"}]


def test_narration_script_runs_immediately_after_story_approval():
    order = []

    def narration_script(state):
        order.append("narration_script")
        return fake_narration_script_creator(state)

    def voice(state):
        order.append("elevenlabs")
        return fake_narration_creator(state)

    def scenes(state):
        order.append("scene_planning")
        return fake_scene_planner(state)

    graph = build_fake_graph(
        narration_script_creator=narration_script,
        narration_creator=voice,
        scene_planner=scenes,
    )
    assert "create_production_bible" not in graph.get_graph().nodes
    config = {"configurable": {"thread_id": "narration-order"}}
    graph.invoke(state(), config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    assert order[:3] == ["narration_script", "elevenlabs", "scene_planning"]


def test_character_references_are_created_after_image_prompt_review():
    order = []

    def prompts(state):
        assert not state.get("character_reference_files")
        order.append("image_prompts")
        return fake_image_prompt_builder(state)

    def references(state):
        assert state.get("image_prompt_requests")
        order.append("character_sheets")
        return fake_reference_creator(state)

    def images(state):
        assert state.get("character_reference_files")
        order.append("shot_images")
        return fake_image_creator(state)

    graph = build_fake_graph(
        image_prompt_builder=prompts,
        reference_creator=references,
        image_creator=images,
    )
    config = {"configurable": {"thread_id": "reference-order"}}
    graph.invoke(state(), config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    assert order == ["image_prompts", "character_sheets", "shot_images"]


def test_failed_shot_image_qa_retries_before_storyboard_review():
    calls = {"images": 0, "qa": 0}

    def images(_state):
        calls["images"] += 1
        return {
            "image_files": [f"outputs/test/images/shot-001-v{calls['images']}.png"],
            "generated_images": [{
                "shot_id": "shot-001", "generation_status": "success",
                "image_path": f"outputs/test/images/shot-001-v{calls['images']}.png",
            }],
            "visual_feedback": [],
        }

    def qa(_state):
        calls["qa"] += 1
        passed = calls["qa"] == 2
        return {
            "shot_image_qa_results": [{
                "shot_id": "shot-001", "approved": passed, "animation_ready": passed,
                "issues": [] if passed else [{"type": "generation_artifact"}],
                "retry_target": None if passed else "image_generation",
            }],
            "shot_image_qa_retry_shots": [] if passed else ["shot-001"],
            "shot_image_qa_round": calls["qa"],
        }

    graph = build_fake_graph(image_creator=images, shot_image_reviewer=qa)
    config = {"configurable": {"thread_id": "shot-image-qa-loop"}}
    graph.invoke(state(), config)
    result = graph.invoke(Command(resume={"action": "approve"}), config)

    assert calls == {"images": 2, "qa": 2}
    assert result["__interrupt__"][0].value["shot_image_qa_results"][0]["approved"] is True


def test_valid_video_routes_through_combined_judge():
    calls = {"validator": 0, "compiler": 0, "judge": 0}

    def validator(_state):
        calls["validator"] += 1
        return {"video_retry_shots": [], "video_validation_exhausted_shots": []}

    def judge(_state):
        calls["judge"] += 1
        return fake_combined_video_judge(_state)

    def compiler(_state):
        calls["compiler"] += 1
        return fake_rough_video_compiler(_state)

    graph = build_fake_graph(
        video_validator=validator, rough_video_compiler=compiler, video_judge=judge,
    )
    config = {"configurable": {"thread_id": "validator-pass"}}
    graph.invoke(state(), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    result = graph.invoke(Command(resume={"action": "approve"}), config)

    assert calls == {"validator": 1, "compiler": 1, "judge": 1}
    assert result["pipeline_status"] == "combined_video_judge_accepted"
    assert result["video_file"].endswith("final_reel.mp4")


def test_narration_revision_returns_to_story_review():
    def narration(_state):
        return {
            "needs_story_revision": True,
            "story_revision_reason": "The ending contradicts the previous beat.",
            "review_note": "The ending contradicts the previous beat.",
        }

    graph = build_fake_graph(narration_script_creator=narration)
    config = {"configurable": {"thread_id": "narration-story-revision"}}
    graph.invoke(state(), config)
    revised = graph.invoke(Command(resume={"action": "approve"}), config)

    assert "story" in revised["__interrupt__"][0].value
    assert "The ending contradicts the previous beat." in revised["story"]


def test_long_audio_regenerates_narration_before_scene_planning():
    calls = {"script": 0, "voice": 0, "scenes": 0}

    def narration_script(state):
        calls["script"] += 1
        return {"narration_script": f"Narration version {calls['script']}.", "narration_feedback": ""}

    def voice(_state):
        calls["voice"] += 1
        if calls["voice"] == 1:
            return {"narration_feedback": "Shorten the narration."}
        return {"narration_file": "narration.mp3", "actual_narration_seconds": 9.5}

    def scenes(state):
        calls["scenes"] += 1
        assert state["narration_script"] == "Narration version 2."
        return fake_scene_planner(state)

    graph = build_fake_graph(
        narration_script_creator=narration_script,
        narration_creator=voice,
        scene_planner=scenes,
    )
    config = {"configurable": {"thread_id": "narration-duration-revision"}}
    graph.invoke(state(), config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    assert calls == {"script": 2, "voice": 2, "scenes": 1}


def test_failed_scene_plan_pauses_for_human_retry():
    calls = {"scenes": 0}

    def scenes(state):
        calls["scenes"] += 1
        if calls["scenes"] == 1:
            return {
                "scenes": [],
                "scene_analysis": [],
                "scene_plan_needs_revision": True,
                "scene_plan_revision_reason": "Narration timing cannot support the proposed boundary.",
                "scene_plan_issues": ["scene-001 ends outside segment-001 timing."],
            }
        return {
            **fake_scene_planner(state),
            "scene_plan_needs_revision": False,
            "scene_plan_revision_reason": None,
            "scene_plan_issues": [],
            "scene_plan_feedback": "",
        }

    graph = build_fake_graph(scene_planner=scenes)
    config = {"configurable": {"thread_id": "scene-plan-human-review"}}
    graph.invoke(state(), config)
    failed = graph.invoke(Command(resume={"action": "approve"}), config)

    review = failed["__interrupt__"][0].value
    assert review["stage"] == "scene_plan_review"
    assert review["actions"] == ["retry", "revise_narration"]

    retried = graph.invoke(Command(resume={"action": "retry", "note": "Use the supplied segment boundary."}), config)
    assert "image_files" in retried["__interrupt__"][0].value
    assert calls == {"scenes": 2}


def test_failed_visual_plan_pauses_for_human_retry():
    calls = {"visual": 0}

    def visual(state):
        calls["visual"] += 1
        if calls["visual"] == 1:
            return {
                "visual_beats": [],
                "visual_plan_needs_revision": True,
                "visual_plan_revision_reason": "A required scene action is missing.",
                "visual_plan_issues": ["scene-001 has no visible action."],
            }
        return {
            **fake_visual_beat_creator(state),
            "visual_plan_needs_revision": False,
            "visual_plan_revision_reason": None,
            "visual_plan_issues": [],
            "visual_plan_feedback": "",
        }

    graph = build_fake_graph(visual_beat_creator=visual)
    config = {"configurable": {"thread_id": "visual-plan-human-review"}}
    graph.invoke(state(), config)
    failed = graph.invoke(Command(resume={"action": "approve"}), config)

    review = failed["__interrupt__"][0].value
    assert review["stage"] == "visual_plan_review"
    assert review["actions"] == ["retry", "revise_scenes"]

    retried = graph.invoke(Command(resume={"action": "retry", "note": "Represent the approved action."}), config)
    assert "image_files" in retried["__interrupt__"][0].value
    assert calls == {"visual": 2}


def test_failed_shot_plan_pauses_for_human_retry():
    calls = {"shots": 0}

    def shots(_state):
        calls["shots"] += 1
        if calls["shots"] == 1:
            return {
                "shot_plan": [],
                "shot_plan_needs_revision": True,
                "shot_plan_revision_reason": "The visual beat contains too many sequential actions for one keyframe.",
                "shot_plan_issues": ["vb-001 cannot be represented as a still image."],
            }
        return {
            **fake_shot_planner(_state),
            "shot_plan_needs_revision": False,
            "shot_plan_revision_reason": None,
            "shot_plan_issues": [],
            "shot_plan_feedback": "",
        }

    graph = build_fake_graph(shot_planner=shots)
    config = {"configurable": {"thread_id": "shot-plan-human-review"}}
    graph.invoke(state(), config)
    failed = graph.invoke(Command(resume={"action": "approve"}), config)

    review = failed["__interrupt__"][0].value
    assert review["stage"] == "shot_plan_review"
    assert review["actions"] == ["retry", "revise_visuals"]

    retried = graph.invoke(Command(resume={"action": "retry", "note": "Split the visible action."}), config)
    assert "image_files" in retried["__interrupt__"][0].value
    assert calls == {"shots": 2}


def test_selected_visual_feedback():
    graph = build_fake_graph()
    config = {"configurable": {"thread_id": "selected-feedback"}}
    graph.invoke(state(), config)
    visual = graph.invoke(Command(resume={"action": "approve"}), config)
    assert "image_files" in visual["__interrupt__"][0].value
    revised_visual = graph.invoke(
        Command(resume={"action": "regenerate", "feedback": [{"shot_id": "shot-001", "note": "warmer light"}]}),
        config,
    )
    assert revised_visual["image_files"][0].endswith("-revised.png")
    assert revised_visual["visual_feedback_history"][0]["revision"] == 1


def test_sqlite_checkpoint_resumes_after_graph_rebuild(tmp_path):
    checkpoint = tmp_path / "reviews.sqlite"
    config = {"configurable": {"thread_id": "restart-resume"}}
    build_fake_graph(agent_graph.persistent_checkpointer(checkpoint)).invoke(state(), config)

    rebuilt = build_fake_graph(agent_graph.persistent_checkpointer(checkpoint))
    assert rebuilt.get_state(config).next == ("review_story",)
    resumed = rebuilt.invoke(Command(resume={"action": "approve"}), config)
    assert "image_files" in resumed["__interrupt__"][0].value
    assert resumed["character_board_file"].endswith("character.png")


def test_technical_video_failure_retries_generation_not_motion_planning():
    calls = {"motion": 0, "video": 0, "validator": 0, "judge": 0}

    def motion(state):
        calls["motion"] += 1
        return fake_motion_planner(state)

    def video_creator(_state):
        calls["video"] += 1
        return {"scene_video_files": [f"video-{calls['video']}.mp4"]}

    def validator(_state):
        calls["validator"] += 1
        return {
            "video_retry_shots": ["shot-001"] if calls["validator"] == 1 else [],
            "video_validation_exhausted_shots": [],
        }

    def judge(_state):
        calls["judge"] += 1
        return fake_combined_video_judge(_state)

    graph = build_fake_graph(
        motion_planner=motion, scene_video_creator=video_creator,
        video_validator=validator, video_judge=judge,
    )
    config = {"configurable": {"thread_id": "validator-retry"}}
    graph.invoke(state(), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    assert calls == {"motion": 1, "video": 2, "validator": 2, "judge": 1}


def test_video_validation_retry_exhaustion_terminates_before_judging():
    calls = {"judge": 0}

    def validator(_state):
        return {
            "video_retry_shots": [],
            "video_validation_exhausted_shots": ["shot-001"],
            "pipeline_status": "video_validation_failed",
        }

    def judge(_state):
        calls["judge"] += 1
        return {}

    graph = build_fake_graph(
        video_validator=validator, video_judge=judge,
    )
    config = {"configurable": {"thread_id": "validator-exhausted"}}
    graph.invoke(state(), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    result = graph.invoke(Command(resume={"action": "approve"}), config)
    assert calls["judge"] == 0
    assert result["pipeline_status"] == "video_validation_failed"


def test_timeline_compilation_failure_replans_before_judging():
    calls = {"planner": 0, "compiler": 0, "judge": 0}

    def planner(_state):
        calls["planner"] += 1
        return fake_timeline_planner(_state)

    def compiler(_state):
        calls["compiler"] += 1
        if calls["compiler"] == 1:
            return {
                "compilation_status": "failed", "rough_cut_failure_source": "timeline",
                "rough_cut_retry_exhausted": False, "rough_cut_retry_count": 1,
                "video_retry_shots": [],
            }
        return fake_rough_video_compiler(_state)

    def judge(_state):
        calls["judge"] += 1
        return fake_combined_video_judge(_state)

    graph = build_fake_graph(
        video_validator=lambda _state: {"video_retry_shots": [], "video_validation_exhausted_shots": []},
        timeline_planner=planner, rough_video_compiler=compiler, video_judge=judge,
    )
    config = {"configurable": {"thread_id": "compiler-timeline-retry"}}
    graph.invoke(state(), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    assert calls == {"planner": 2, "compiler": 2, "judge": 1}


def test_combined_video_failure_regenerates_video_then_rejudges():
    calls = {"motion": 0, "video": 0, "judge": 0}

    def motion(state):
        calls["motion"] += 1
        return fake_motion_planner(state)

    def video(_state):
        calls["video"] += 1
        return {"scene_video_files": [f"video-{calls['video']}.mp4"]}

    def validator(_state):
        return {"video_retry_shots": [], "video_validation_exhausted_shots": []}

    def judge(_state):
        calls["judge"] += 1
        retry = calls["judge"] == 1
        return {
            "combined_video_judge_approved": not retry,
            "combined_video_judge_retry_target": "video_generation" if retry else "none",
            "video_retry_shots": ["shot-001"] if retry else [],
            "video_judge_refinement_exhausted": False,
            "pipeline_status": "combined_video_judge_retry" if retry else "combined_video_judge_accepted",
        }

    graph = build_fake_graph(
        motion_planner=motion, scene_video_creator=video, video_validator=validator,
        video_judge=judge,
    )
    config = {"configurable": {"thread_id": "meta-video-retry"}}
    graph.invoke(state(), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    result = graph.invoke(Command(resume={"action": "approve"}), config)

    assert calls == {"motion": 1, "video": 2, "judge": 2}
    assert result["pipeline_status"] == "combined_video_judge_accepted"


def test_combined_motion_failure_replans_motion_before_video_retry():
    calls = {"motion": 0, "video": 0, "judge": 0}

    def motion(state):
        calls["motion"] += 1
        return fake_motion_planner(state)

    def video(_state):
        calls["video"] += 1
        return {"scene_video_files": [f"video-{calls['video']}.mp4"]}

    def judge(_state):
        calls["judge"] += 1
        retry = calls["judge"] == 1
        return {
            "combined_video_judge_approved": not retry,
            "combined_video_judge_retry_target": "motion_planner" if retry else "none",
            "motion_plan_retry_shots": ["shot-001"] if retry else [],
            "video_retry_shots": ["shot-001"] if retry else [],
            "video_judge_refinement_exhausted": False,
        }

    graph = build_fake_graph(
        motion_planner=motion, scene_video_creator=video,
        video_validator=lambda _state: {"video_retry_shots": [], "video_validation_exhausted_shots": []},
        video_judge=judge,
    )
    config = {"configurable": {"thread_id": "meta-motion-retry"}}
    graph.invoke(state(), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    assert calls == {"motion": 2, "video": 2, "judge": 2}


if __name__ == "__main__":
    test_character_reference_is_automatic_then_visual_storyboard_is_reviewed()
    test_story_feedback_is_applied_and_recorded()
    test_selected_visual_feedback()
