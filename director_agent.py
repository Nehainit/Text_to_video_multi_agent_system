import math
import re

from models import invoke_with_evaluation, load_model
from prompts import director_plan_prompt, director_revision_prompt, production_bible_prompt
from schema import AgentState
from story_agent import _duration_seconds, _max_words, _parse_json


MOTIONS = {"static", "zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down"}
TRANSITIONS = {"cut", "match_cut", "dissolve"}
AUDIO_BRIDGES = {"none", "j_cut", "l_cut"}
COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _requested_count(topic: str, noun: str) -> int | None:
    numbers = r"\d+|" + "|".join(COUNT_WORDS)
    match = re.search(rf"\b({numbers})\s+{noun}s?\b", topic.lower())
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else COUNT_WORDS[value]


def _narration_words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _count_targets(topic: str, duration: str, story: str = "") -> tuple[int, int, int | None, int | None, str]:
    seconds = _duration_seconds(duration)
    minimum_shots = max(1, math.ceil(seconds / 15))
    maximum_shots = max(1, seconds // 2)
    requested_shots = _requested_count(topic, "shot")
    requested_scenes = _requested_count(topic, "scene")
    notes = []

    if requested_shots is None and story:
        word_count = len(_narration_words(story))
        requested_shots = min(max(math.ceil(word_count / 10), minimum_shots), maximum_shots)
        notes.append(f"Calculated {requested_shots} shots from {word_count} narration words.")
    elif requested_shots is not None:
        adjusted = min(max(requested_shots, minimum_shots), maximum_shots)
        if adjusted != requested_shots:
            notes.append(
                f"Requested {requested_shots} shots; Director adjusted to {adjusted} to preserve the complete story arc."
            )
        requested_shots = adjusted
    if requested_scenes is not None:
        scene_limit = requested_shots or maximum_shots
        adjusted = min(requested_scenes, scene_limit)
        if adjusted != requested_scenes:
            notes.append(f"Requested {requested_scenes} scenes; Director adjusted to {adjusted} so every scene has a shot.")
        requested_scenes = adjusted

    return minimum_shots, maximum_shots, requested_shots, requested_scenes, " ".join(notes)


def _validate_bible(value: object, expected_characters: int | None = None) -> dict:
    required = {"theme", "visual_style", "palette", "lighting", "camera_language", "locations", "characters", "continuity_rules"}
    if not isinstance(value, dict) or not required <= value.keys():
        raise RuntimeError("Director production bible is missing required fields.")
    if not isinstance(value["characters"], list) or not isinstance(value["locations"], list):
        raise RuntimeError("Director production bible characters and locations must be lists.")
    locations = []
    for index, location in enumerate(value["locations"], start=1):
        if isinstance(location, dict):
            description = str(location.get("description") or location.get("name") or "").strip()
            location_id = str(location.get("location_id") or f"location-{index:03}").strip()
        else:
            description, location_id = str(location).strip(), f"location-{index:03}"
        if not description or not location_id:
            raise RuntimeError("Every production bible location needs an ID and description.")
        locations.append({"location_id": location_id, "description": description})
    if not locations:
        raise RuntimeError("Director production bible needs at least one approved location.")
    character_fields = {"character_id", "name", "appearance", "wardrobe", "props"}
    if any(not isinstance(character, dict) or not character_fields <= character.keys() for character in value["characters"]):
        raise RuntimeError("Every production bible character needs an ID, name, appearance, wardrobe, and props.")
    if expected_characters is not None and len(value["characters"]) != expected_characters:
        raise RuntimeError(f"Production bible needs exactly the {expected_characters} characters extracted from the approved story.")
    return {**value, "locations": locations}


def create_production_bible(state: AgentState) -> dict:
    if not state.get("story"):
        raise RuntimeError("Director needs an approved story before creating the production bible.")
    model = load_model("director")
    feedback = (
        f"Current production bible: {state.get('production_bible', {})}. "
        f"Human reference revision: {state['reference_feedback']}. Change only what the feedback requires."
        if state.get("reference_feedback")
        else ""
    )
    error = None
    evaluations = list(state.get("llm_evaluations", []))
    for _ in range(3):
        response, evaluation = invoke_with_evaluation(
            model,
            production_bible_prompt(
                topic=state["topic"],
                story=state["story"],
                story_beats=state.get("story_beats", []),
                characters=state.get("characters", []),
                feedback=feedback if error is None else f"{feedback}\nPrevious attempt failed: {error}",
            ),
            agent_name="director",
            purpose="create_production_bible",
        )
        evaluation["call_number"] = len(evaluations) + 1
        evaluations.append(evaluation)
        try:
            bible = _validate_bible(_parse_json(response.content).get("production_bible"))
            characters = [
                " — ".join(filter(None, (str(item.get("name", "")).strip(), str(item.get("appearance", "")).strip(), str(item.get("wardrobe", "")).strip())))
                for item in bible["characters"]
                if isinstance(item, dict)
            ]
            return {"production_bible": bible, "characters": characters, "llm_evaluations": evaluations}
        except RuntimeError as exc:
            error = exc
    raise error or RuntimeError("Director failed to create the production bible.")


def _fit_durations(shots: list[dict], total_seconds: int) -> list[int]:
    count = len(shots)
    if not 2 * count <= total_seconds <= 15 * count:
        raise RuntimeError("Shot count cannot fit the requested duration using 2-15 second clips.")
    weights = []
    for shot in shots:
        try:
            weights.append(max(0.1, float(shot.get("duration_seconds", 1))))
        except (TypeError, ValueError):
            weights.append(1.0)
    durations = [2] * count
    for _ in range(total_seconds - 2 * count):
        candidates = [index for index, duration in enumerate(durations) if duration < 15]
        index = max(candidates, key=lambda item: weights[item] / durations[item])
        durations[index] += 1
    return durations


def _timestamp(seconds: int) -> str:
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02}:{seconds:02}"


def _finalize_plan(
    plan: object,
    state: AgentState,
    minimum_shots: int,
    maximum_shots: int,
    required_shots: int | None,
    required_scenes: int | None,
    preserve_timing: bool = False,
) -> list[dict]:
    if not isinstance(plan, list) or not plan:
        raise RuntimeError("Director returned no shots.")
    if required_shots is not None and len(plan) != required_shots:
        raise RuntimeError(f"Director plan needs exactly {required_shots} shots.")
    if not minimum_shots <= len(plan) <= maximum_shots:
        raise RuntimeError(f"Director plan needs {minimum_shots}-{maximum_shots} shots for this duration.")

    required = {
        "scene_id",
        "story_purpose",
        "characters_present",
        "duration_seconds",
        "narration",
        "visuals",
        "camera",
        "motion",
        "subject_motion",
        "continuity_in",
        "continuity_out",
        "transition_to_next",
        "sfx_prompt",
    }
    normalized = []
    scene_map = {}
    cast = {}
    for character in state.get("production_bible", {}).get("characters", []):
        if isinstance(character, dict):
            character_id = str(character.get("character_id", "")).strip()
            name = str(character.get("name", "")).strip()
            if character_id:
                cast[character_id.casefold()] = character_id
                if name:
                    cast[name.casefold()] = character_id
    for index, original in enumerate(plan, start=1):
        if not isinstance(original, dict) or not required <= original.keys():
            raise RuntimeError("Director shot is missing required fields.")
        if original["motion"] not in MOTIONS:
            raise RuntimeError("Director shot motion is invalid.")
        if not str(original["narration"]).strip():
            raise RuntimeError("Every shot needs continuous voice-over narration.")
        if len(str(original["camera"]).split()) < 4:
            raise RuntimeError("Every shot needs a cinematic camera plan.")
        if not str(original["continuity_in"]).strip() or not str(original["continuity_out"]).strip():
            raise RuntimeError("Every shot needs entering and leaving continuity state.")
        if not isinstance(original["characters_present"], list):
            raise RuntimeError("Every shot needs a characters_present list, including an empty list for environment-only shots.")
        try:
            characters_present = list(dict.fromkeys(cast[str(value).strip().casefold()] for value in original["characters_present"]))
        except KeyError as exc:
            raise RuntimeError(f"Director introduced a character outside the approved cast: {exc.args[0]}") from exc

        source_scene = str(original["scene_id"])
        scene_map.setdefault(source_scene, f"scene-{len(scene_map) + 1:03}")
        transition = original["transition_to_next"]
        if not isinstance(transition, dict) or transition.get("type") not in TRANSITIONS:
            raise RuntimeError("Director transition type is invalid.")
        if transition.get("audio_bridge", "none") not in AUDIO_BRIDGES:
            raise RuntimeError("Director audio bridge is invalid.")
        transition_seconds = float(transition.get("duration_seconds", 0))
        if transition["type"] != "dissolve":
            transition_seconds = 0.0
        elif not 0.25 <= transition_seconds <= 1.0:
            raise RuntimeError("Dissolve duration must be between 0.25 and 1 second.")

        shot = {
            **original,
            "shot_id": f"shot-{index:03}",
            "shot_number": index,
            "scene_number": index,
            "scene_id": scene_map[source_scene],
            "characters_present": characters_present,
            "transition_to_next": {
                "type": transition["type"],
                "duration_seconds": transition_seconds,
                "audio_bridge": transition.get("audio_bridge", "none"),
            },
        }
        normalized.append(shot)

    if required_scenes is not None and len(scene_map) != required_scenes:
        raise RuntimeError(f"Director plan needs exactly {required_scenes} narrative scenes.")
    for previous, current in zip(normalized, normalized[1:]):
        if previous["scene_id"] == current["scene_id"]:
            current["continuity_in"] = previous["continuity_out"]
    if len(normalized) == 1:
        if str(normalized[0]["story_purpose"]).lower() != "opening_resolution":
            raise RuntimeError("A one-shot film must establish and resolve the story.")
    else:
        if str(normalized[0]["story_purpose"]).lower() != "opening":
            raise RuntimeError("First shot must be the opening.")
        if str(normalized[-1]["story_purpose"]).lower() != "resolution":
            raise RuntimeError("Last shot must resolve the story.")
    narration = " ".join(str(shot["narration"]) for shot in normalized)
    narration_words = len(_narration_words(narration))
    if narration_words > _max_words(state["duration"]):
        raise RuntimeError("Director narration is too long for continuous voice-over at this duration.")
    if narration_words < max(8, round(_duration_seconds(state["duration"]))):
        raise RuntimeError("Director narration is too short to carry the requested film duration.")
    approved_narration = state.get("narration_script") or state.get("story", "")
    if approved_narration and _narration_words(narration) != _narration_words(approved_narration):
        raise RuntimeError("Director must split the approved story verbatim across shots without adding, removing, or reordering narration.")

    total_seconds = _duration_seconds(state["duration"])
    if preserve_timing:
        if sum(int(shot["duration_seconds"]) for shot in normalized) != total_seconds:
            raise RuntimeError("Revised Director plan changed the approved duration.")
        return normalized

    cursor = 0
    timing_weights = [{**shot, "duration_seconds": len(_narration_words(str(shot["narration"])))} for shot in normalized]
    for shot, duration in zip(normalized, _fit_durations(timing_weights, total_seconds)):
        shot["duration_seconds"] = duration
        shot["start_time"] = _timestamp(cursor)
        cursor += duration
        shot["end_time"] = _timestamp(cursor)
    normalized[-1]["transition_to_next"] = {"type": "cut", "duration_seconds": 0.0, "audio_bridge": "none"}
    return normalized


def create_director_plan(state: AgentState) -> dict:
    if not state.get("production_bible") or not state.get("story"):
        raise RuntimeError("Director needs an approved story and reference package.")
    minimum, maximum, requested_shots, requested_scenes, count_adjustment = _count_targets(
        state["topic"], state["duration"], state["story"]
    )
    feedback = state.get("director_feedback", [])
    current = state.get("director_plan") or state.get("storyboard")
    model = load_model("director")
    error = None

    for _ in range(3):
        if feedback and current:
            response = model.invoke(
                director_revision_prompt(plan=current, feedback=feedback, production_bible=state["production_bible"])
            )
            proposed = _parse_json(response.content).get("director_plan")
            if not isinstance(proposed, list):
                error = RuntimeError("Director revision returned no plan.")
                continue
            proposed_by_id = {item.get("shot_id"): item for item in proposed if isinstance(item, dict)}
            selected = {item["shot_id"] for item in feedback}
            if not selected <= proposed_by_id.keys():
                error = RuntimeError("Director revision omitted a selected shot.")
                continue
            merged = []
            for shot in current:
                revised = proposed_by_id.get(shot["shot_id"], shot) if shot["shot_id"] in selected else shot
                merged.append(
                    {
                        **revised,
                        **{key: shot[key] for key in ("shot_id", "scene_id", "scene_number", "shot_number", "duration_seconds", "start_time", "end_time")},
                    }
                )
            candidate = merged
            preserve_timing = True
            plan_minimum = plan_maximum = len(current)
            plan_shots = len(current)
            plan_scenes = len({shot["scene_id"] for shot in current})
        else:
            response = model.invoke(
                director_plan_prompt(
                    topic=state["topic"],
                    story=state["story"],
                    story_beats=state.get("story_beats", []),
                    production_bible=state["production_bible"],
                    duration_seconds=_duration_seconds(state["duration"]),
                    minimum_shots=minimum,
                    maximum_shots=maximum,
                    required_shots=requested_shots,
                    required_scenes=requested_scenes,
                    feedback="" if error is None else f"Previous attempt failed: {error}",
                )
            )
            candidate = _parse_json(response.content).get("director_plan")
            preserve_timing = False
            plan_minimum, plan_maximum, plan_shots, plan_scenes = minimum, maximum, requested_shots, requested_scenes
        try:
            plan = _finalize_plan(
                candidate,
                state,
                plan_minimum,
                plan_maximum,
                plan_shots,
                plan_scenes,
                preserve_timing,
            )
            return {
                "director_plan": plan,
                "storyboard": plan,
                "director_approved": False,
                "director_feedback": [],
                "last_director_feedback": feedback,
                "count_adjustment": count_adjustment,
            }
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            error = RuntimeError(str(exc))
    raise error or RuntimeError("Director failed to create a valid shot plan.")
