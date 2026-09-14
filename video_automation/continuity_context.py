from typing import Any


def character_definitions(state: dict[str, Any]) -> list[dict[str, str]]:
    """Return deterministic character IDs from approved user requirements."""
    legacy = state.get("production_bible") or {}
    if legacy.get("characters"):
        return [item for item in legacy["characters"] if isinstance(item, dict)]

    values = (
        (state.get("story_outline") or {}).get("characters")
        or (state.get("parsed_requirements") or {}).get("characters")
        or state.get("characters")
        or []
    )
    if isinstance(values, str):
        values = [item.strip() for item in values.split(",") if item.strip()]
    result = []
    for index, value in enumerate(values, start=1):
        if isinstance(value, dict):
            name = str(value.get("name", "")).strip()
            description = str(value.get("description") or name).strip()
        else:
            name = description = str(value).strip()
        if name:
            result.append({
                "character_id": f"character-{index:03}",
                "name": name,
                "description": description,
            })
    return result


def location_definitions(state: dict[str, Any]) -> list[dict[str, str]]:
    """Return locations already approved in the scene plan."""
    legacy = state.get("production_bible") or {}
    if legacy.get("locations"):
        return [item for item in legacy["locations"] if isinstance(item, dict)]

    locations = {}
    for scene in state.get("scenes") or []:
        location_id = str(scene.get("location_id", "")).strip()
        if location_id and location_id not in locations:
            description = str(scene.get("location_description") or scene.get("location") or "").strip()
            if not description:
                description = "; ".join(str(item) for item in scene.get("visible_actions", []) if str(item).strip())
            locations[location_id] = {"location_id": location_id, "description": description or location_id}
    return list(locations.values())


def continuity_context(state: dict[str, Any]) -> dict[str, Any]:
    requirements = state.get("parsed_requirements") or {}
    return {
        "visual_style": requirements.get("visual_style") or state.get("visual_style"),
        "characters": character_definitions(state),
        "locations": location_definitions(state),
    }
