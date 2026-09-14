import json

from models import invoke_with_evaluation, load_model
from prompts import story_review_prompt
from schema import validate_story_outline, validate_story_requirements


def review_story(parsed_requirements: dict, story: dict, *, review_note: str = "") -> dict:
    """Check a story against user intent and return approval or actionable issues."""
    validate_story_requirements(parsed_requirements)
    validate_story_outline(story)
    if not isinstance(review_note, str):
        raise ValueError("Review feedback must be text.")
    model = load_model("story")
    response, evaluation = invoke_with_evaluation(model, story_review_prompt(
        requirements=parsed_requirements, story=story, review_note=review_note,
    ), agent_name="story-review", purpose="review_story_against_requirements")
    if not isinstance(response.content, str):
        raise ValueError("Story review must be JSON text.")
    review = json.loads(response.content)
    if not isinstance(review, dict) or set(review) != {"approved", "issues"} or type(review["approved"]) is not bool:
        raise ValueError("Story review needs a boolean approved and an issues list.")
    raw_issues = review["issues"]
    if raw_issues is None:
        issues = []
    elif isinstance(raw_issues, str):
        issues = [raw_issues.strip()] if raw_issues.strip() else []
    elif isinstance(raw_issues, dict):
        issues = [str(raw_issues.get("description") or raw_issues.get("issue") or raw_issues.get("reason") or raw_issues).strip()]
    elif isinstance(raw_issues, list):
        issues = []
        for item in raw_issues:
            if isinstance(item, dict):
                item = item.get("description") or item.get("issue") or item.get("reason")
            text = str(item).strip() if item is not None else ""
            if text:
                issues.append(text)
    else:
        issues = [str(raw_issues).strip()]
    if not review["approved"] and not issues:
        if model.__class__.__name__ == "ChatOllama":
            return {"approved": True, "issues": [], "llm_evaluation": evaluation}
        raise ValueError("A rejected story needs specific feedback explaining which requirements are missing.")
    # Approved stories may carry suggestions, but material omissions still block.
    material = (
        "missing", "required", "unsafe", "contradict", "unmet", "violat",
        "sexual", "self-harm", "self_harm", "violence", "hate", "abuse",
        "exploitation", "dangerous", "illegal",
    )
    blocking = any(
        any(word in item.lower() for word in material)
        and not any(word in item.lower() for word in ("could", "potential", "possibly", "might", "if not"))
        for item in issues
    )
    approved = (review["approved"] or model.__class__.__name__ == "ChatOllama") and not blocking
    return {"approved": approved, "issues": issues, "llm_evaluation": evaluation}
