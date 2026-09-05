import json
import re

from models import load_model
from prompts import story_prompt
from schema import AgentState


def _duration_seconds(duration: str) -> int:
    match = re.search(r"\d+(?:\.\d+)?", duration)
    if not match:
        return 30
    seconds = float(match.group())
    if "min" in duration.lower():
        seconds *= 60
    return max(5, int(round(seconds)))


def _max_words(duration: str) -> int:
    return max(30, min(90, round(_duration_seconds(duration) * 2.5)))


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    return json.loads(text)


def _sentences(story: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?।])\s+", story) if sentence.strip()]


def _characters_list(characters: str | list[str]) -> list[str]:
    if isinstance(characters, str):
        return [character.strip() for character in characters.split(",") if character.strip()]
    return characters


def _characters_text(characters: str | list[str]) -> str:
    if isinstance(characters, str):
        return characters
    return ", ".join(characters)


def _validate_story(story: str, state: AgentState) -> None:
    lowered = story.lower()
    if any(word in lowered for word in ("time:", "visuals:", "frames:", "scene 1")):
        raise RuntimeError("Story agent returned scene planning text instead of only a story.")
    for character in _characters_list(state["characters"]):
        if len(character.split()) == 1 and re.search(rf"\b{re.escape(character)}\s*,\s+an?\s+", story, re.IGNORECASE):
            raise RuntimeError("Story agent invented a new character identity.")

    sentences = _sentences(story)
    if len(sentences) < 3:
        raise RuntimeError("Story needs at least start, turn, and landing sentences.")
    if len(story.split()) > _max_words(state["duration"]):
        raise RuntimeError("Story is too long for the requested duration.")
    if len(sentences[-1].split()) < 6:
        raise RuntimeError("Story ending is too abrupt.")


def create_story(state: AgentState) -> dict[str, str]:
    model = load_model("story")
    max_words = _max_words(state["duration"])
    bare_names = [character for character in _characters_list(state["characters"]) if len(character.split()) == 1]
    identity_rule = ""
    if bare_names:
        identity_rule = (
            "Do not describe bare names with invented identities. "
            f"Forbidden patterns: {', '.join(f'{name}, a ...' for name in bare_names)}."
        )
    review_note = f'Human revision note: {state["review_note"]}' if state.get("review_note") else ""
    feedback = ""
    for attempt in range(3):
        response = model.invoke(
            story_prompt(
                topic=state["topic"],
                tone=state["tone"],
                duration=state["duration"],
                language=state["language"],
                characters=_characters_text(state["characters"]),
                max_words=max_words,
                identity_rule=identity_rule,
                review_note=review_note,
                feedback=feedback,
            )
        )
        data = _parse_json(response.content)
        if any(key in data for key in ("frames", "scenes", "storyboard", "image_prompts")):
            error = RuntimeError("Story agent returned planning data instead of only a story.")
        else:
            story = str(data["story"]).strip()
            if not story:
                error = RuntimeError("Story agent returned an empty story.")
            else:
                try:
                    _validate_story(story, state)
                    return {"story": story}
                except RuntimeError as exc:
                    error = exc
        if attempt == 2:
            raise error
        feedback = f"Previous attempt failed: {error}. Fix that and return only the JSON story."

    raise RuntimeError("Story agent failed.")
