def story_prompt(
    *,
    topic: str,
    tone: str,
    duration: str,
    language: str,
    characters: str,
    max_words: int,
    identity_rule: str,
    review_note: str,
    feedback: str,
) -> str:
    return f"""
Return JSON only: {{"story": "..."}}.
Create only the story text, not frames, not scenes, not timestamps, not image prompts.
Topic: {topic}
Tone: {tone}
Duration: {duration}
Language: {language}
Characters: {characters}
Maximum words: {max_words}.
Use 3 or 4 sentences only:
1. Start: place the characters in the situation.
2. Turn: something goes wrong, changes, or creates a choice.
3. Landing ending: resolve the choice and make the final feeling clear.
4. Optional final beat only if needed.
Treat character entries as fixed names/descriptions; do not change them or invent new identities.
{identity_rule}
{review_note}
The story value must be a single narrative paragraph.
{feedback}
"""


def storyboard_prompt(*, duration: str, scene_count: int, story: str, review_note: str, feedback: str) -> str:
    return f"""
Return JSON only: {{"storyboard": [...]}}.
Create a video storyboard from this story.
Duration: {duration}
Scene count: exactly {scene_count}
Story: {story}

Each storyboard item must include:
- scene_number: integer
- start_time: timestamp like "00:00"
- end_time: timestamp like "00:12"
- visuals: concrete cinematic keyframe description for image generation, including location, characters, action, mood, lighting, and composition
- camera: cinematic camera plan with shot size, angle, lens/framing, composition, and movement
- motion: one of static, zoom_in, zoom_out, pan_left, pan_right, tilt_up, tilt_down
- narration: matching story/narration line
- sfx_prompt: short non-speech sound effects prompt for this exact scene

Keep scenes in chronological order and cover the full story arc.
The camera field must agree with motion, e.g. if motion is zoom_in, camera should describe what the zoom pushes toward.
Use varied cinematic camera movement across scenes. Avoid repeating the same camera plan unless the story needs it.
{review_note}
If a human revision request is present, regenerate the full storyboard and make the requested changes obvious in the relevant scenes.
{feedback}
"""


def image_prompt(scene: dict, review_note: str = "") -> str:
    return (
        f"Cinematic keyframe for scene {scene['scene_number']}: {scene['visuals']} "
        f"Storyboard camera plan: {scene['camera']}. "
        f"Editor motion to support: {scene.get('motion', 'static')}. "
        "Compose the still image so this camera movement will look natural in video. "
        "Vertical 9:16 cinematic story frame, expressive lighting, depth, consistent characters, "
        "no text, no watermark, no logo. "
        f"{f'Human visual revision request: {review_note}' if review_note else ''}"
    )


def mood_board_prompt(*, story: str, tone: str, characters: str | list[str], storyboard: list[dict]) -> str:
    visual_arc = " ".join(str(scene["visuals"]) for scene in storyboard)
    character_text = characters if isinstance(characters, str) else ", ".join(characters)
    return (
        f"Cinematic mood board for this story: {story} Tone: {tone}. Characters: {character_text}. "
        f"Visual arc: {visual_arc}. Create a cohesive production mood board with color palette, lighting, "
        "locations, textures, wardrobe, and cinematic atmosphere. No text, no watermark, no logo."
    )


def character_sheet_prompt(characters: list[str]) -> str:
    character_text = ", ".join(characters)
    return (
        f"Clean combined character reference sheet for: {character_text}. "
        "Show each character separately in the same image, full-body front view, "
        "consistent face, outfit, hairstyle, and body shape. Plain white background, "
        "clear spacing between characters, no text, no watermark, no logo."
    )
