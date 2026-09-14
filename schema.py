from typing import Any, Literal, NotRequired, TypedDict


def validate_story_requirements(requirements: object) -> None:
    text_fields = {"topic", "language", "tone", "visual_style"}
    list_fields = {"characters", "must_include", "must_avoid", "other_constraints"}
    if not isinstance(requirements, dict) or set(requirements) != text_fields | list_fields | {"duration_seconds"}:
        raise ValueError("parsed_requirements must contain exactly the nine specified fields.")
    for key in text_fields:
        value = requirements[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"parsed_requirements.{key} must be nonempty text or null.")
    if requirements["topic"] is None:
        raise ValueError("parsed_requirements.topic is required.")
    for key in list_fields:
        value = requirements[key]
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"parsed_requirements.{key} must be a list of nonempty strings.")
    duration = requirements["duration_seconds"]
    if duration is not None and (type(duration) is not int or duration <= 0):
        raise ValueError("duration_seconds must be a positive integer or null.")


def validate_story_outline(story: object) -> None:
    if not isinstance(story, dict) or set(story) not in ({"idea", "structure"}, {"idea", "characters", "structure"}):
        raise ValueError("story must contain only idea, characters, and structure.")
    if not isinstance(story["idea"], str) or not story["idea"].strip():
        raise ValueError("story.idea must be nonempty text.")
    characters = story.get("characters", [])
    if not isinstance(characters, list):
        raise ValueError("story.characters must be a list.")
    for index, character in enumerate(characters, start=1):
        if not isinstance(character, dict) or set(character) != {"character_id", "name", "description"}:
            raise ValueError("Each character must contain only character_id, name, and description.")
        if character["character_id"] != f"character-{index:03}":
            raise ValueError("character_id values must be consecutive: character-001, character-002, etc.")
        if any(not isinstance(character[key], str) or not character[key].strip() for key in ("name", "description")):
            raise ValueError("Every character needs a nonempty name and description.")
    beats = story["structure"]
    if not isinstance(beats, list) or len(beats) < 3:
        raise ValueError("story.structure needs at least three beats: setup, meaningful change, and resolution.")
    for index, beat in enumerate(beats, start=1):
        if not isinstance(beat, dict) or set(beat) != {"beat_id", "description"}:
            raise ValueError("Each beat must contain only beat_id and description.")
        if beat["beat_id"] != f"beat-{index:03}":
            raise ValueError("beat_id values must be consecutive: beat-001, beat-002, etc.")
        if not isinstance(beat["description"], str) or not beat["description"].strip():
            raise ValueError("Each beat description must be nonempty text.")


class VideoCandidate(TypedDict):
    shot_id: str
    candidate_id: str
    round: int
    prompt: str
    seed: int
    file: str | None


class VideoValidationIssue(TypedDict):
    type: str
    description: str
    expected: NotRequired[Any]
    actual: NotRequired[Any]


class VideoValidationMetadata(TypedDict):
    duration_seconds: float | None
    width: int | None
    height: int | None
    aspect_ratio: float | None
    fps: float | None
    frame_count: int | None
    video_codec: str | None
    has_video_stream: bool
    has_audio_stream: bool
    file_size_bytes: int


class VideoValidationResult(TypedDict):
    shot_id: str
    valid: bool
    issues: list[VideoValidationIssue]
    metadata: VideoValidationMetadata


class EditTimelineItem(TypedDict):
    timeline_item_id: str
    shot_id: str
    scene_id: str
    video_file: str
    source_in: float
    source_out: float
    timeline_start: float
    timeline_end: float
    narration_segment_ids: list[str]
    visual_beat_ids: list[str]
    transition_in: str
    transition_out: str
    edit_reason: str


class EditTimelineIssue(TypedDict):
    type: str
    shot_id: str
    required_duration_seconds: float
    actual_usable_duration_seconds: float
    description: str


class EditTimelineRetryRequest(TypedDict):
    shot_id: str
    retry_target: Literal["video_generation", "motion_planner", "none"]
    reason: str


class ShotTimestamp(TypedDict):
    timeline_item_id: str
    shot_id: str
    scene_id: str
    start_sec: float
    end_sec: float
    narration_segment_ids: list[str]


class CompilationIssue(TypedDict):
    type: str
    description: str
    shot_id: NotRequired[str]
    expected: NotRequired[Any]
    actual: NotRequired[Any]


class CombinedVideoJudgeScores(TypedDict):
    visual_fidelity: int
    character_consistency: int
    motion_quality: int
    temporal_consistency: int
    shot_alignment: int
    narration_alignment: int
    story_faithfulness: int
    emotional_intent: int
    pacing_and_editing: int
    overall_coherence: int


class CombinedVideoJudgeIssue(TypedDict):
    type: str
    severity: Literal["minor", "major", "critical"]
    start_sec: float | None
    end_sec: float | None
    shot_ids: list[str]
    description: str
    failure_source: Literal["source_image", "motion_plan", "video_generation", "edit_timeline", "none"]
    retry_target: Literal["image_generation", "motion_planner", "video_generation", "plan_edit_timeline", "none"]
    correction: str


class CombinedVideoJudgeReport(TypedDict):
    approved: bool
    overall_score: float
    scores: CombinedVideoJudgeScores
    issues: list[CombinedVideoJudgeIssue]
    problem_shot_ids: list[str]
    primary_failure_source: Literal["source_image", "motion_plan", "video_generation", "edit_timeline", "none"]
    retry_target: Literal["image_generation", "motion_planner", "video_generation", "plan_edit_timeline", "none"]
    refinement_needed: bool
    summary: str


VisualFidelityMetric = Literal[
    "character_identity",
    "anatomy_integrity",
    "character_scale_consistency",
    "composition_preservation",
    "framing_preservation",
    "location_consistency",
    "style_consistency",
    "object_persistence",
    "visual_artifact_control",
    "temporal_visual_consistency",
]


class VisualFidelityScores(TypedDict):
    character_identity: int
    anatomy_integrity: int
    character_scale_consistency: int
    composition_preservation: int
    framing_preservation: int
    location_consistency: int
    style_consistency: int
    object_persistence: int
    visual_artifact_control: int
    temporal_visual_consistency: int


class VisualFidelityIssue(TypedDict):
    metric: VisualFidelityMetric
    severity: Literal["minor", "major", "critical"]
    description: str
    timestamp_or_range: str | None


class VisualFidelityReport(TypedDict):
    judge: Literal["visual_fidelity"]
    scores: VisualFidelityScores
    issues: list[VisualFidelityIssue]


class JudgeIssue(TypedDict):
    metric: str
    severity: Literal["minor", "major", "critical"]
    description: str
    timestamp_or_range: str | None


class MotionTemporalScores(TypedDict):
    subject_motion_alignment: int
    camera_motion_alignment: int
    environment_motion_alignment: int
    motion_naturalness: int
    motion_intensity_alignment: int
    physical_plausibility: int
    identity_stability_over_time: int
    background_stability: int
    temporal_consistency: int
    pacing: int


class MotionTemporalReport(TypedDict):
    judge: Literal["motion_temporal"]
    scores: MotionTemporalScores
    issues: list[JudgeIssue]


class ContextIntentScores(TypedDict):
    shot_alignment: int
    narration_alignment: int
    story_faithfulness: int
    emotional_intent: int
    shot_purpose: int
    semantic_coherence: int
    action_faithfulness: int
    character_role_faithfulness: int
    absence_of_unsupported_events: int


class ContextIntentReport(TypedDict):
    judge: Literal["context_intent"]
    scores: ContextIntentScores
    issues: list[JudgeIssue]


class AdversarialReport(TypedDict):
    challenges: list[dict[str, Any]]
    missed_issues: list[dict[str, Any]]


class MetaJudgeReport(TypedDict):
    approved: bool
    overall_score: float
    final_scores: dict[str, int]
    strong_metrics: list[str]
    weak_metrics: list[str]
    critical_issues: list[str]
    failure_source: Literal["source_image", "motion_plan", "video_generation", "none"]
    retry_target: Literal["image_generation", "motion_planner", "video_generation", "none"]
    refinement_needed: bool
    refinement_brief: list[str]


class AgentState(TypedDict):
    thread_id: NotRequired[str]
    topic: str
    tone: str
    duration: str
    language: str
    characters: str | list[str]
    genre: NotRequired[str]
    genre_structure: NotRequired[list[str]]
    output_dir: NotRequired[str]
    quality_mode: NotRequired[Literal["standard", "refine"]]
    aspect_ratio: NotRequired[Literal["16:9", "9:16", "1:1"]]
    video_quality: NotRequired[Literal["standard", "high"]]
    story: NotRequired[str]
    story_outline: NotRequired[dict[str, Any]]
    parsed_requirements: NotRequired[dict[str, Any]]
    llm_evaluations: NotRequired[list[dict[str, Any]]]
    planning_attempts: NotRequired[dict[str, dict[str, Any]]]
    story_beats: NotRequired[list[dict[str, Any]]]
    narration_script: NotRequired[str]
    narration_segments: NotRequired[list[dict[str, Any]]]
    estimated_narration_seconds: NotRequired[float]
    needs_story_revision: NotRequired[bool]
    story_revision_reason: NotRequired[str | None]
    scene_analysis: NotRequired[list[dict[str, Any]]]
    scenes: NotRequired[list[dict[str, Any]]]
    scene_plan_needs_revision: NotRequired[bool]
    scene_plan_revision_reason: NotRequired[str | None]
    scene_plan_issues: NotRequired[list[str]]
    scene_plan_feedback: NotRequired[str]
    visual_beats: NotRequired[list[dict[str, Any]]]
    visual_plan_needs_revision: NotRequired[bool]
    visual_plan_revision_reason: NotRequired[str | None]
    visual_plan_issues: NotRequired[list[str]]
    visual_plan_feedback: NotRequired[str]
    shot_plan: NotRequired[list[dict[str, Any]]]
    shot_plan_needs_revision: NotRequired[bool]
    shot_plan_revision_reason: NotRequired[str | None]
    shot_plan_issues: NotRequired[list[str]]
    shot_plan_feedback: NotRequired[str]
    image_prompt_requests: NotRequired[list[dict[str, Any]]]
    critic_issues: NotRequired[list[str]]
    reference_approved: NotRequired[bool]
    reference_feedback: NotRequired[str]
    last_reference_feedback: NotRequired[str]
    reference_feedback_history: NotRequired[list[dict[str, Any]]]
    character_board_file: NotRequired[str]
    storyboard: NotRequired[list[dict[str, Any]]]
    director_plan: NotRequired[list[dict[str, Any]]]
    director_approved: NotRequired[bool]
    director_feedback: NotRequired[list[dict[str, Any]]]
    last_director_feedback: NotRequired[list[dict[str, Any]]]
    director_feedback_history: NotRequired[list[dict[str, Any]]]
    count_adjustment: NotRequired[str]
    mood_board_file: NotRequired[str]
    character_reference_files: NotRequired[list[str]]
    image_files: NotRequired[list[str | None]]
    generated_images: NotRequired[list[dict[str, Any]]]
    shot_image_qa_results: NotRequired[list[dict[str, Any]]]
    shot_image_qa_retry_shots: NotRequired[list[str]]
    shot_image_qa_pending_shots: NotRequired[list[str]]
    shot_image_qa_round: NotRequired[int]
    motion_plans: NotRequired[list[dict[str, Any]]]
    motion_plan_needs_revision: NotRequired[bool]
    motion_plan_revision_reason: NotRequired[str | None]
    motion_plan_retry_shots: NotRequired[list[str]]
    motion_plan_retry_feedback: NotRequired[dict[str, str]]
    generated_videos: NotRequired[list[dict[str, Any]]]
    scene_video_files: NotRequired[list[str | None]]
    video_candidates: NotRequired[list[VideoCandidate]]
    video_refinement_round: NotRequired[int]
    video_retry_shots: NotRequired[list[str]]
    video_revised_prompts: NotRequired[dict[str, str]]
    video_validation_results: NotRequired[list[VideoValidationResult]]
    video_validation_retry_counts: NotRequired[dict[str, int]]
    video_validation_exhausted_shots: NotRequired[list[str]]
    edit_timeline: NotRequired[list[EditTimelineItem]]
    narration_duration_seconds: NotRequired[float]
    timeline_duration_seconds: NotRequired[float]
    timeline_valid: NotRequired[bool]
    timeline_issues: NotRequired[list[EditTimelineIssue]]
    timeline_retry_requests: NotRequired[list[EditTimelineRetryRequest]]
    edit_timeline_retry_counts: NotRequired[dict[str, int]]
    edit_timeline_exhausted_shots: NotRequired[list[str]]
    compilation_status: NotRequired[Literal["success", "failed"]]
    rough_cut_file: NotRequired[str | None]
    duration_seconds: NotRequired[float | None]
    width: NotRequired[int | None]
    height: NotRequired[int | None]
    fps: NotRequired[float | None]
    shot_timestamps: NotRequired[list[ShotTimestamp]]
    sync_valid: NotRequired[bool]
    compilation_issues: NotRequired[list[CompilationIssue]]
    compilation_error: NotRequired[str | None]
    rough_cut_failure_source: NotRequired[Literal["timeline", "source_video", "runtime"] | None]
    rough_cut_retry_count: NotRequired[int]
    rough_cut_retry_exhausted: NotRequired[bool]
    combined_video_judge_report: NotRequired[CombinedVideoJudgeReport | None]
    combined_video_judge_approved: NotRequired[bool]
    combined_video_judge_status: NotRequired[Literal["accepted", "retry", "error", "exhausted"]]
    combined_video_judge_error: NotRequired[str | None]
    combined_video_judge_error_count: NotRequired[int]
    combined_video_judge_error_exhausted: NotRequired[bool]
    combined_video_judge_model: NotRequired[str]
    combined_video_judge_retry_target: NotRequired[Literal["image_generation", "motion_planner", "video_generation", "plan_edit_timeline", "none"]]
    combined_video_judge_retry_shots: NotRequired[list[str]]
    video_judge_refinement_round: NotRequired[int]
    video_judge_refinement_exhausted: NotRequired[bool]
    visual_fidelity_reports: NotRequired[dict[str, VisualFidelityReport]]
    motion_temporal_reports: NotRequired[dict[str, MotionTemporalReport]]
    context_intent_reports: NotRequired[dict[str, ContextIntentReport]]
    adversarial_reports: NotRequired[dict[str, AdversarialReport]]
    meta_judge_reports: NotRequired[dict[str, MetaJudgeReport]]
    judge_retry_counts: NotRequired[dict[str, int]]
    judge_retry_shots: NotRequired[list[str]]
    judge_retry_target: NotRequired[str | None]
    judge_exhausted_shots: NotRequired[list[str]]
    judge_decision: NotRequired[Literal["ACCEPT", "REFINE", "REGENERATE"]]
    pipeline_status: NotRequired[str]
    narration_file: NotRequired[str]
    narration_alignment_file: NotRequired[str]
    narration_alignment: NotRequired[dict[str, Any]]
    narration_segment_timings: NotRequired[list[dict[str, float | int | str]]]
    actual_narration_seconds: NotRequired[float]
    narration_feedback: NotRequired[str]
    scene_timings: NotRequired[list[dict[str, float | int]]]
    sfx_files: NotRequired[list[str]]
    music_file: NotRequired[str]
    sound_design_plan: NotRequired[list[dict[str, Any]]]
    subtitles: NotRequired[list[dict[str, Any]]]
    subtitle_file: NotRequired[str]
    mixed_audio_file: NotRequired[str]
    video_file: NotRequired[str]
    elevenlabs_voice_id: NotRequired[str]
    elevenlabs_tts_model: NotRequired[str]
    elevenlabs_sfx_model: NotRequired[str]
    approved: NotRequired[bool]
    review_note: NotRequired[str]
    last_story_feedback: NotRequired[str]
    story_feedback_history: NotRequired[list[dict[str, Any]]]
    visual_approved: NotRequired[bool]
    visual_feedback: NotRequired[list[dict[str, Any]]]
    last_visual_feedback: NotRequired[list[dict[str, Any]]]
    visual_feedback_history: NotRequired[list[dict[str, Any]]]
    warnings: NotRequired[list[str]]
