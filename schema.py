from typing import Any, NotRequired, TypedDict


class AgentState(TypedDict):
    topic: str
    tone: str
    duration: str
    language: str
    characters: str | list[str]
    output_dir: NotRequired[str]
    story: NotRequired[str]
    storyboard: NotRequired[list[dict[str, Any]]]
    mood_board_file: NotRequired[str]
    character_reference_files: NotRequired[list[str]]
    image_files: NotRequired[list[str]]
    scene_video_files: NotRequired[list[str]]
    narration_file: NotRequired[str]
    narration_alignment_file: NotRequired[str]
    scene_timings: NotRequired[list[dict[str, float | int]]]
    sfx_files: NotRequired[list[str]]
    subtitles: NotRequired[list[dict[str, Any]]]
    subtitle_file: NotRequired[str]
    mixed_audio_file: NotRequired[str]
    video_file: NotRequired[str]
    elevenlabs_voice_id: NotRequired[str]
    elevenlabs_tts_model: NotRequired[str]
    elevenlabs_sfx_model: NotRequired[str]
    approved: NotRequired[bool]
    review_note: NotRequired[str]
    storyboard_approved: NotRequired[bool]
    storyboard_review_note: NotRequired[str]
    visual_approved: NotRequired[bool]
    visual_feedback: NotRequired[list[dict[str, Any]]]
    last_visual_feedback: NotRequired[list[dict[str, Any]]]
    visual_feedback_history: NotRequired[list[dict[str, Any]]]
