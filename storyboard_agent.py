from schema import AgentState
from models import load_model
from prompts import storyboard_prompt
from story_agent import _duration_seconds, _parse_json

MOTIONS = {"static", "zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down"}


def _scene_count(duration: str) -> int:
    return max(3, min(8, round(_duration_seconds(duration) / 12)))


def _timestamp_seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


def _validate_storyboard(storyboard: object, scene_count: int, duration: str) -> None:
    if not isinstance(storyboard, list) or not storyboard:
        raise RuntimeError("Storyboard agent returned no scenes.")
    if len(storyboard) != scene_count:
        raise RuntimeError(f"Storyboard needs exactly {scene_count} scenes.")

    required = {"scene_number", "start_time", "end_time", "visuals", "camera", "motion", "narration"}
    previous_end = 0.0
    for index, scene in enumerate(storyboard, start=1):
        if not isinstance(scene, dict) or not required <= scene.keys():
            raise RuntimeError("Storyboard scene is missing required fields.")
        if scene["scene_number"] != index:
            raise RuntimeError("Storyboard scene numbers must be sequential.")
        if scene["motion"] not in MOTIONS:
            raise RuntimeError("Storyboard scene motion is invalid.")
        if len(str(scene["camera"]).split()) < 4:
            raise RuntimeError("Storyboard camera must describe a cinematic shot plan.")
        start = _timestamp_seconds(str(scene["start_time"]))
        end = _timestamp_seconds(str(scene["end_time"]))
        if start < previous_end or end <= start:
            raise RuntimeError("Storyboard timestamps must be increasing.")
        previous_end = end

    if abs(previous_end - _duration_seconds(duration)) > 2:
        raise RuntimeError("Storyboard must cover the requested duration.")


def create_storyboard(state: AgentState) -> dict[str, list[dict]]:
    story = state.get("story")
    if not story:
        raise RuntimeError("Storyboard agent needs an approved story.")

    model = load_model("storyboard")
    scene_count = _scene_count(state["duration"])
    feedback = ""
    for attempt in range(3):
        response = model.invoke(
            storyboard_prompt(
                duration=state["duration"],
                scene_count=scene_count,
                story=story,
                review_note=(
                    f"Human revision request: {state['storyboard_review_note']}"
                    if state.get("storyboard_review_note")
                    else ""
                ),
                feedback=feedback,
            )
        )
        storyboard = _parse_json(response.content).get("storyboard")
        try:
            _validate_storyboard(storyboard, scene_count, state["duration"])
            return {"storyboard": storyboard}
        except RuntimeError as exc:
            if attempt == 2:
                raise
            feedback = f"Previous attempt failed: {exc}. Fix it and return valid JSON only."

    raise RuntimeError("Storyboard agent failed.")
