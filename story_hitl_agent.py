from schema import AgentState
from langgraph.types import interrupt

def review_story(state: AgentState):
    decision = interrupt(
        {
            "story": state["story"],
            "actions": ["approve", "edit", "regenerate"],
        }
    )

    action = decision.get("action", "approve")

    if action == "edit":
        return {
            "story": decision["story"],
            "approved": False,
            "review_note": "",
        }

    if action == "regenerate":
        return {
            "approved": False,
            "review_note": decision.get("note") or "Regenerate the story.",
        }

    return {"approved": True}


def review_storyboard(state: AgentState):
    decision = interrupt(
        {
            "storyboard": state["storyboard"],
            "actions": ["approve", "regenerate"],
            "feedback": state.get("storyboard_review_note", ""),
        }
    )

    if decision.get("action") == "regenerate":
        return {
            "storyboard_approved": False,
            "storyboard_review_note": decision.get("note") or "Regenerate the storyboard.",
        }

    return {"storyboard_approved": True}


def review_visual_storyboard(state: AgentState):
    decision = interrupt(
        {
            "storyboard": state["storyboard"],
            "mood_board_file": state["mood_board_file"],
            "image_files": state["image_files"],
            "actions": ["approve", "regenerate"],
            "feedback": state.get("last_visual_feedback", []),
        }
    )

    if decision.get("action") == "regenerate":
        feedback = decision.get("feedback") or []
        known_scenes = {scene["scene_number"] for scene in state["storyboard"]}
        if not feedback or any(item.get("scene_number") not in known_scenes or not str(item.get("note", "")).strip() for item in feedback):
            raise RuntimeError("Visual feedback needs a valid scene number and note.")
        return {
            "visual_approved": False,
            "visual_feedback": feedback,
            "visual_feedback_history": [
                *state.get("visual_feedback_history", []),
                {"revision": len(state.get("visual_feedback_history", [])) + 1, "feedback": feedback},
            ],
        }

    return {"visual_approved": True, "visual_feedback": []}
