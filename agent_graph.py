import sqlite3
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph

from combined_video_judge import combined_video_judge
from editor_agent import edit_video
from edit_timeline_agent import plan_edit_timeline
from image_agent import build_image_prompts, create_reference_package, create_visual_storyboard
from motion_planner_agent import create_motion_plans
from narration_agent import create_narration
from preproduction_agents import create_narration_script, create_visual_beats, critique_shots, plan_scenes, plan_shots
from prompts import SHOT_IMAGE_QA_CONFIG
from rough_video_compiler import compile_rough_video
from schema import AgentState
from shot_image_qa_agent import review_shot_images
from shot_video_generation_agent import create_shot_videos
from story_agent import create_story_node
from story_hitl_agent import review_scene_plan, review_shot_plan, review_story, review_visual_plan, review_visual_storyboard
from subtitle_agent import create_subtitles
from video_validator import validate_generated_videos


def persistent_checkpointer(path: Path):
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError:
        # ponytail: memory fallback keeps development running until the optional SQLite package is installed.
        return InMemorySaver()
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def route_combined_video_judge(state: AgentState) -> str:
    if state.get("combined_video_judge_approved"):
        return "create_subtitles"
    if state.get("combined_video_judge_error_exhausted") or state.get("video_judge_refinement_exhausted"):
        return END
    return {
        "image_generation": "create_storyboard",
        "motion_planner": "plan_motion",
        "video_generation": "create_scene_videos",
        "plan_edit_timeline": "plan_edit_timeline",
    }.get(state.get("combined_video_judge_retry_target"), "combined_video_judge")


def _logged_step(name: str, action: Callable[[AgentState], dict]):
    def run(state: AgentState) -> dict:
        prefix = f"[pipeline:{state.get('thread_id', '-')}]"
        print(f"{prefix} START {name}", flush=True)
        try:
            result = action(state)
        except GraphInterrupt:
            print(f"{prefix} PAUSED {name}", flush=True)
            raise
        except Exception as exc:
            print(f"{prefix} ERROR {name}: {type(exc).__name__}: {exc}", flush=True)
            raise
        print(f"{prefix} DONE {name}", flush=True)
        return result

    return run


def build_story_graph(
    *,
    story_creator: Callable[[AgentState], dict] = create_story_node,
    reference_creator: Callable[[AgentState], dict] = create_reference_package,
    narration_script_creator: Callable[[AgentState], dict] = create_narration_script,
    scene_planner: Callable[[AgentState], dict] = plan_scenes,
    visual_beat_creator: Callable[[AgentState], dict] = create_visual_beats,
    shot_planner: Callable[[AgentState], dict] = plan_shots,
    director_critic: Callable[[AgentState], dict] = critique_shots,
    image_prompt_builder: Callable[[AgentState], dict] = build_image_prompts,
    image_creator: Callable[[AgentState], dict] = create_visual_storyboard,
    shot_image_reviewer: Callable[[AgentState], dict] = review_shot_images,
    motion_planner: Callable[[AgentState], dict] = create_motion_plans,
    narration_creator: Callable[[AgentState], dict] = create_narration,
    scene_video_creator: Callable[[AgentState], dict] = create_shot_videos,
    soundfx_creator: Callable[[AgentState], dict] | None = None,  # retained for caller compatibility; stage removed
    video_validator: Callable[[AgentState], dict] | None = validate_generated_videos,
    timeline_planner: Callable[[AgentState], dict] = plan_edit_timeline,
    rough_video_compiler: Callable[[AgentState], dict] = compile_rough_video,
    video_judge: Callable[[AgentState], dict] = combined_video_judge,
    sound_plan_creator: Callable[[AgentState], dict] | None = None,  # retained for caller compatibility; stage removed
    subtitle_creator: Callable[[AgentState], dict] = create_subtitles,
    editor: Callable[[AgentState], dict] = edit_video,
    checkpointer=None,
):
    def route_story(state: AgentState) -> str:
        if state.get("approved"):
            return "create_narration_script"
        if state.get("review_note"):
            return "create_story"
        return "review_story"

    def route_narration_script(state: AgentState) -> str:
        if state.get("needs_story_revision"):
            return "create_story"
        return "create_narration"

    def route_narration_audio(state: AgentState) -> str:
        if state.get("narration_feedback"):
            return "create_narration_script"
        return "plan_scenes"

    def route_scene_plan(state: AgentState) -> str:
        return "review_scene_plan" if state.get("scene_plan_needs_revision") else "create_visual_beats"

    def route_scene_review(state: AgentState) -> str:
        return "create_narration_script" if state.get("narration_feedback") else "plan_scenes"

    def route_visual_plan(state: AgentState) -> str:
        return "review_visual_plan" if state.get("visual_plan_needs_revision") else "plan_shots"

    def route_visual_plan_review(state: AgentState) -> str:
        return "plan_scenes" if state.get("scene_plan_feedback") else "create_visual_beats"

    def route_shot_plan(state: AgentState) -> str:
        return "review_shot_plan" if state.get("shot_plan_needs_revision") else "critique_shots"

    def route_shot_plan_review(state: AgentState) -> str:
        return "create_visual_beats" if state.get("visual_plan_feedback") else "plan_shots"

    def route_shot_critique(state: AgentState) -> str:
        if state.get("shot_plan_needs_revision"):
            return "review_shot_plan"
        return "build_image_prompts"

    def route_visual(state: AgentState) -> str:
        if state.get("visual_approved"):
            return "plan_motion"
        if state.get("visual_feedback"):
            return "create_storyboard"
        return "review_visual_storyboard"

    def route_shot_image_qa(state: AgentState) -> str:
        if (
            state.get("shot_image_qa_retry_shots")
            and state.get("shot_image_qa_round", 0) <= int(SHOT_IMAGE_QA_CONFIG["max_retries"])
        ):
            return "create_storyboard"
        return "review_visual_storyboard"

    def route_video_validation(state: AgentState) -> str:
        if state.get("video_validation_exhausted_shots"):
            return END
        return "create_scene_videos" if state.get("video_retry_shots") else "plan_edit_timeline"

    def route_edit_timeline(state: AgentState) -> str:
        if state.get("edit_timeline_exhausted_shots"):
            return END
        return "create_scene_videos" if state.get("video_retry_shots") else "compile_rough_video"

    def route_rough_compilation(state: AgentState) -> str:
        if state.get("compilation_status") == "success":
            return "combined_video_judge"
        if state.get("rough_cut_retry_exhausted"):
            return END
        return {
            "timeline": "plan_edit_timeline",
            "source_video": "create_scene_videos",
            "runtime": "compile_rough_video",
        }.get(state.get("rough_cut_failure_source"), END)

    def route_motion(state: AgentState) -> str:
        return "review_shot_plan" if state.get("motion_plan_needs_revision") else "create_scene_videos"

    graph = StateGraph(AgentState)
    graph.add_node("create_story", _logged_step("create_story", story_creator))
    graph.add_node("review_story", _logged_step("review_story", review_story))
    graph.add_node("create_reference_package", _logged_step("create_reference_package", reference_creator))
    graph.add_node("create_narration_script", _logged_step("create_narration_script", narration_script_creator))
    graph.add_node("plan_scenes", _logged_step("plan_scenes", scene_planner))
    graph.add_node("review_scene_plan", _logged_step("review_scene_plan", review_scene_plan))
    graph.add_node("create_visual_beats", _logged_step("create_visual_beats", visual_beat_creator))
    graph.add_node("review_visual_plan", _logged_step("review_visual_plan", review_visual_plan))
    graph.add_node("plan_shots", _logged_step("plan_shots", shot_planner))
    graph.add_node("review_shot_plan", _logged_step("review_shot_plan", review_shot_plan))
    graph.add_node("critique_shots", _logged_step("critique_shots", director_critic))
    graph.add_node("build_image_prompts", _logged_step("build_image_prompts", image_prompt_builder))
    graph.add_node("create_storyboard", _logged_step("create_storyboard", image_creator))
    graph.add_node("review_shot_images", _logged_step("review_shot_images", shot_image_reviewer))
    graph.add_node("plan_motion", _logged_step("plan_motion", motion_planner))
    graph.add_node("review_visual_storyboard", _logged_step("review_visual_storyboard", review_visual_storyboard))
    graph.add_node("create_narration", _logged_step("create_narration", narration_creator))
    graph.add_node("create_scene_videos", _logged_step("create_scene_videos", scene_video_creator))
    graph.add_node("create_subtitles", _logged_step("create_subtitles", subtitle_creator))
    graph.add_node("edit_video", _logged_step("edit_video", editor))

    graph.add_edge(START, "create_story")
    graph.add_edge("create_story", "review_story")
    graph.add_conditional_edges("review_story", route_story)
    graph.add_conditional_edges("create_narration_script", route_narration_script)
    graph.add_conditional_edges("create_narration", route_narration_audio)
    graph.add_conditional_edges("plan_scenes", route_scene_plan)
    graph.add_conditional_edges("review_scene_plan", route_scene_review)
    graph.add_conditional_edges("create_visual_beats", route_visual_plan)
    graph.add_conditional_edges("review_visual_plan", route_visual_plan_review)
    graph.add_conditional_edges("plan_shots", route_shot_plan)
    graph.add_conditional_edges("review_shot_plan", route_shot_plan_review)
    graph.add_edge("create_storyboard", "review_shot_images")
    graph.add_conditional_edges("review_shot_images", route_shot_image_qa)
    graph.add_conditional_edges("review_visual_storyboard", route_visual)
    graph.add_conditional_edges("plan_motion", route_motion)
    graph.add_edge("build_image_prompts", "create_reference_package")
    graph.add_edge("create_reference_package", "create_storyboard")
    graph.add_conditional_edges("critique_shots", route_shot_critique)
    if video_validator:
        graph.add_node("validate_videos", _logged_step("validate_videos", video_validator))
        graph.add_node("plan_edit_timeline", _logged_step("plan_edit_timeline", timeline_planner))
        graph.add_node("compile_rough_video", _logged_step("compile_rough_video", rough_video_compiler))
        graph.add_node("combined_video_judge", _logged_step("combined_video_judge", video_judge))
        graph.add_edge("create_scene_videos", "validate_videos")
        graph.add_conditional_edges("validate_videos", route_video_validation)
        graph.add_conditional_edges("plan_edit_timeline", route_edit_timeline)
        graph.add_conditional_edges("compile_rough_video", route_rough_compilation)
        graph.add_conditional_edges("combined_video_judge", route_combined_video_judge)
    else:
        graph.add_edge("create_scene_videos", "create_subtitles")
    graph.add_edge("create_subtitles", "edit_video")
    graph.add_edge("edit_video", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
