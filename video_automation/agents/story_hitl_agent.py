from video_automation.schema import AgentState
from langgraph.types import interrupt

def review_story(state: AgentState):
    decision = interrupt(
        {
            "stage": "story_review",
            "story": state["story"],
            "actions": ["approve", "edit", "regenerate"],
            "feedback": state.get("last_story_feedback", ""),
        }
    )

    action = decision.get("action", "approve")

    if action == "edit":
        return {
            "story": decision["story"].strip(),
            "story_outline": {},
            "story_beats": [],
            "characters": [],
            "approved": True,
            "review_note": "",
        }

    if action == "regenerate":
        note = str(decision.get("note", "")).strip()
        if not note:
            raise RuntimeError("Story feedback is required.")
        return {
            "approved": False,
            "review_note": note,
            "last_story_feedback": note,
            "story_feedback_history": [
                *state.get("story_feedback_history", []),
                {"revision": len(state.get("story_feedback_history", [])) + 1, "note": note},
            ],
            "story_beats": [],
            "characters": [],
        }

    return {"approved": True}


def review_scene_plan(state: AgentState):
    decision = interrupt(
        {
            "stage": "scene_plan_review",
            "revision_reason": state.get("scene_plan_revision_reason"),
            "issues": state.get("scene_plan_issues", []),
            "actions": ["retry", "revise_narration"],
        }
    )
    action = decision.get("action", "retry")
    note = str(decision.get("note") or state.get("scene_plan_revision_reason") or "Revise the scene plan.").strip()
    if action == "revise_narration":
        return {
            "narration_feedback": note,
            "scene_plan_feedback": "",
            "scene_plan_needs_revision": False,
        }
    return {
        "scene_plan_feedback": note,
        "scene_plan_needs_revision": False,
    }


def review_visual_plan(state: AgentState):
    decision = interrupt(
        {
            "stage": "visual_plan_review",
            "revision_reason": state.get("visual_plan_revision_reason"),
            "issues": state.get("visual_plan_issues", []),
            "actions": ["retry", "revise_scenes"],
        }
    )
    action = decision.get("action", "retry")
    note = str(decision.get("note") or state.get("visual_plan_revision_reason") or "Revise the visual plan.").strip()
    if action == "revise_scenes":
        return {
            "scene_plan_feedback": note,
            "visual_plan_feedback": "",
            "visual_plan_needs_revision": False,
        }
    return {
        "visual_plan_feedback": note,
        "visual_plan_needs_revision": False,
    }


def review_shot_plan(state: AgentState):
    decision = interrupt(
        {
            "stage": "shot_plan_review",
            "revision_reason": state.get("shot_plan_revision_reason"),
            "issues": state.get("shot_plan_issues", []),
            "actions": ["retry", "revise_visuals"],
        }
    )
    note = str(decision.get("note") or state.get("shot_plan_revision_reason") or "Revise the shot plan.").strip()
    if decision.get("action") == "revise_visuals":
        return {
            "visual_plan_feedback": note,
            "shot_plan_feedback": "",
            "shot_plan_needs_revision": False,
        }
    return {"shot_plan_feedback": note, "shot_plan_needs_revision": False}


def review_reference_package(state: AgentState):
    decision = interrupt(
        {
            "stage": "reference_board_review",
            "production_bible": state["production_bible"],
            "character_board_file": state["character_board_file"],
            "mood_board_file": state["mood_board_file"],
            "actions": ["approve", "regenerate"],
            "feedback": state.get("last_reference_feedback", ""),
        }
    )

    if decision.get("action") == "regenerate":
        note = str(decision.get("note", "")).strip()
        if not note:
            raise RuntimeError("Reference package feedback is required.")
        return {
            "reference_approved": False,
            "reference_feedback": note,
            "reference_feedback_history": [
                *state.get("reference_feedback_history", []),
                {"revision": len(state.get("reference_feedback_history", [])) + 1, "note": note},
            ],
        }

    return {"reference_approved": True, "reference_feedback": ""}


def review_director_plan(state: AgentState):
    decision = interrupt(
        {
            "stage": "director_plan_review",
            "director_plan": state["director_plan"],
            "count_adjustment": state.get("count_adjustment", ""),
            "actions": ["approve", "regenerate"],
            "feedback": state.get("last_director_feedback", []),
        }
    )

    if decision.get("action") == "regenerate":
        feedback = decision.get("feedback") or []
        known_shots = {shot["shot_id"] for shot in state["director_plan"]}
        shot_ids = [item.get("shot_id") for item in feedback]
        if (
            not feedback
            or len(shot_ids) != len(set(shot_ids))
            or any(shot_id not in known_shots or not str(item.get("note", "")).strip() for shot_id, item in zip(shot_ids, feedback))
        ):
            raise RuntimeError("Director feedback needs one valid note per selected shot.")
        return {
            "director_approved": False,
            "director_feedback": feedback,
            "director_feedback_history": [
                *state.get("director_feedback_history", []),
                {"revision": len(state.get("director_feedback_history", [])) + 1, "feedback": feedback},
            ],
        }

    return {"director_approved": True, "director_feedback": []}


def review_visual_storyboard(state: AgentState):
    decision = interrupt(
        {
            "stage": "visual_storyboard_review",
            "storyboard": state["storyboard"],
            "mood_board_file": state["mood_board_file"],
            "image_files": state["image_files"],
            "shot_image_qa_results": state.get("shot_image_qa_results", []),
            "actions": ["approve", "regenerate"],
            "feedback": state.get("last_visual_feedback", []),
        }
    )

    if decision.get("action") == "regenerate":
        feedback = decision.get("feedback") or []
        known_shots = {shot["shot_id"] for shot in state["storyboard"]}
        shot_ids = [item.get("shot_id") for item in feedback]
        if (
            not feedback
            or len(shot_ids) != len(set(shot_ids))
            or any(shot_id not in known_shots or not str(item.get("note", "")).strip() for shot_id, item in zip(shot_ids, feedback))
        ):
            raise RuntimeError("Visual feedback needs one valid note per selected shot.")
        return {
            "visual_approved": False,
            "visual_feedback": feedback,
            "shot_image_qa_retry_shots": [],
            "shot_image_qa_round": 0,
            "visual_feedback_history": [
                *state.get("visual_feedback_history", []),
                {"revision": len(state.get("visual_feedback_history", [])) + 1, "feedback": feedback},
            ],
        }

    return {"visual_approved": True, "visual_feedback": []}
