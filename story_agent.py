import json
import re
from copy import deepcopy

from models import invoke_with_evaluation, load_model
from prompts import CONTENT_SAFETY_SYSTEM_PROMPT, SAFETY_CATEGORIES, story_prompt
from schema import AgentState, validate_story_outline, validate_story_requirements
from story_review_agent import review_story


class UnsafeContentError(ValueError):
    def __init__(self, categories: list[str]):
        self.categories = categories
        super().__init__(f"Request blocked by content safety: {', '.join(categories)}.")


def _duration_seconds(duration: str) -> int:
    match = re.search(r"\d+(?:\.\d+)?", duration)
    if not match:
        return 30
    seconds = float(match.group())
    if "min" in duration.lower():
        seconds *= 60
    return max(5, int(round(seconds)))


def _max_words(duration: str) -> int:
    return max(8, round(_duration_seconds(duration) * 2))


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    return json.loads(text)


def _complete_story_outline(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"idea", "characters", "structure"}:
        raise ValueError("story must contain only idea, characters, and structure.")
    if not isinstance(value["characters"], list):
        raise ValueError("story.characters must be a list.")
    characters = []
    for index, character in enumerate(value["characters"], start=1):
        if not isinstance(character, dict) or not set(character) <= {"character_id", "name", "description"} or not {"name", "description"} <= set(character):
            raise ValueError("Each generated character must contain only name and description.")
        if any(not isinstance(character[key], str) or not character[key].strip() for key in ("name", "description")):
            raise ValueError("Every generated character needs a nonempty name and description.")
        characters.append({
            "character_id": f"character-{index:03}",
            "name": character["name"].strip(),
            "description": character["description"].strip(),
        })
    if not isinstance(value["structure"], list):
        raise ValueError("story.structure must be a list.")
    beats = []
    for index, beat in enumerate(value["structure"], start=1):
        if not isinstance(beat, dict) or not set(beat) <= {"beat_id", "description"} or "description" not in beat:
            raise ValueError("Each generated story beat must contain only description.")
        beats.append({"beat_id": f"beat-{index:03}", "description": beat["description"]})
    return {"idea": value["idea"], "characters": characters, "structure": beats}


def _check_input_safety(content: dict) -> dict:
    model = load_model("safety")
    response, evaluation = invoke_with_evaluation(
        model,
        [
            {"role": "system", "content": CONTENT_SAFETY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
        ],
        agent_name="safety",
        purpose="review_user_input_safety",
    )
    if not isinstance(response.content, str):
        raise RuntimeError("Input safety control must return JSON text.")
    try:
        result = _parse_json(response.content)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Input safety control returned invalid JSON.") from exc
    if not isinstance(result, dict) or not {"safe", "categories", "reason"} <= set(result):
        raise RuntimeError("Input safety control returned an invalid decision.")
    safe, categories, reason = result["safe"], result["categories"], result["reason"]
    if isinstance(safe, str) and safe.strip().casefold() in {"true", "false"}:
        safe = safe.strip().casefold() == "true"
    if isinstance(categories, str):
        try:
            parsed_categories = json.loads(categories)
        except (TypeError, ValueError):
            parsed_categories = [categories.strip()] if categories.strip() else []
        categories = parsed_categories
    if isinstance(reason, str) and reason.strip().casefold() in {"", "none", "null", "n/a"}:
        reason = None
    if safe is True and isinstance(reason, str):
        reason = None
    if safe is True and isinstance(categories, list):
        blocked_categories = [category for category in categories if category in SAFETY_CATEGORIES]
        if blocked_categories:
            categories = blocked_categories
        else:
            categories = []
    if (
        type(safe) is not bool
        or not isinstance(categories, list)
        or any(category not in SAFETY_CATEGORIES for category in categories)
        or len(categories) != len(set(categories))
        or (safe and (categories or reason is not None))
        or (not safe and (not categories or not isinstance(reason, str) or not reason.strip()))
    ):
        raise RuntimeError("Input safety control returned an invalid decision.")
    if not safe:
        raise UnsafeContentError(categories)
    return evaluation


def create_story(user_input: str | dict) -> dict:
    """Return the reviewed story outline and parsed requirements, without producing media."""
    if isinstance(user_input, str):
        user_input = {"topic": user_input}
    if not isinstance(user_input, dict) or not isinstance(user_input.get("topic"), str) or not user_input["topic"].strip():
        raise ValueError("A nonempty user request or topic is required.")
    user_request = {
        key: user_input[key]
        for key in ("topic", "duration", "language", "tone", "characters", "visual_style", "must_include", "must_avoid", "other_constraints")
        if key in user_input
    }
    review_note = user_input.get("review_note", "")
    original_story = user_input.get("story_outline") or user_input.get("story")
    previous_story = original_story
    requirements = deepcopy(user_input.get("parsed_requirements"))
    if requirements is not None:
        validate_story_requirements(requirements)
    feedback = ""
    safety_content = user_request if requirements is None else ({"review_note": review_note} if review_note else None)
    evaluations = [_check_input_safety(safety_content)] if safety_content else []
    model = load_model("story")
    for attempt in range(3):
        response, evaluation = invoke_with_evaluation(model, story_prompt(
            user_request=user_request if requirements is None else None,
            requirements=requirements, previous_story=previous_story,
            review_note=review_note, feedback=feedback,
        ), agent_name="story", purpose="generate_story_and_requirements" if requirements is None else "regenerate_story")
        evaluations.append(evaluation)
        try:
            if not isinstance(response.content, str):
                raise ValueError("Story response must be JSON text.")
            result = _parse_json(response.content)
            expected_fields = {"story", "parsed_requirements"} if requirements is None else {"story"}
            if not isinstance(result, dict) or set(result) != expected_fields:
                raise ValueError(
                    "Return exactly story and parsed_requirements." if requirements is None
                    else "Return only story; parsed requirements are already saved and must not be generated again."
                )
            if requirements is None:
                validate_story_requirements(result["parsed_requirements"])
                requirements = result["parsed_requirements"]
            result = {"story": _complete_story_outline(result["story"]), "parsed_requirements": requirements}
            previous_story = result["story"]
            validate_story_outline(previous_story)
            story_text = " ".join(beat["description"] for beat in previous_story["structure"])
            if review_note and (previous_story == original_story or story_text == original_story):
                raise ValueError("Apply the human revision note instead of returning the unchanged story.")
            review = review_story(requirements, previous_story, review_note=review_note)
            evaluations.append(review.pop("llm_evaluation"))
            if review["approved"]:
                for number, item in enumerate(evaluations, start=1):
                    item["call_number"] = number
                return {**result, "llm_evaluations": evaluations}
            feedback = "; ".join(review["issues"])
        except (ValueError, TypeError, KeyError) as exc:
            feedback = str(exc)
    raise RuntimeError(f"Story failed validation or internal review after 3 attempts: {feedback}")


def create_story_node(state: AgentState) -> dict:
    """Adapt the story-only contract to the existing video workflow's prose input."""
    result = create_story(state)
    beats = result["story"]["structure"]
    return {
        "story": " ".join(beat["description"] for beat in beats),
        "story_outline": result["story"],
        "story_beats": beats,
        "parsed_requirements": result["parsed_requirements"],
        "llm_evaluations": result["llm_evaluations"],
        "characters": result["story"]["characters"],
        "review_note": "",
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate a reviewed story idea, structure, and parsed requirements as JSON.")
    parser.add_argument("prompt", nargs="?", help="User request; reads standard input when omitted.")
    args = parser.parse_args()
    if args.prompt is not None:
        user_request = args.prompt
    else:
        if sys.stdin.isatty():
            print("Describe your story requirements:", file=sys.stderr)
            user_request = input()
        else:
            user_request = sys.stdin.read()
    try:
        print(json.dumps(create_story(user_request), indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
