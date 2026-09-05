from typing import Callable

from langgraph.graph import StateGraph ,START,END
from schema import AgentState
from story_agent import create_story
from storyboard_agent import create_storyboard
from image_agent import create_scene_videos_node, create_visual_storyboard
from narration_agent import create_narration
from soundfx_agent import create_soundfx
from subtitle_agent import create_subtitles
from editor_agent import edit_video
from story_hitl_agent import review_story, review_storyboard, review_visual_storyboard
from langgraph.checkpoint.memory import InMemorySaver

def build_story_graph(
    story_creator: Callable[[AgentState], dict] = create_story,
    storyboard_creator: Callable[[AgentState], dict] = create_storyboard,
    image_creator: Callable[[AgentState], dict] = create_visual_storyboard,
    narration_creator: Callable[[AgentState], dict] = create_narration,
    soundfx_creator: Callable[[AgentState], dict] = create_soundfx,
    subtitle_creator: Callable[[AgentState], dict] = create_subtitles,
    editor: Callable[[AgentState], dict] = edit_video,
    scene_video_creator: Callable[[AgentState], dict] = create_scene_videos_node,
):

    def route_after_review(state:AgentState) -> str:
        if state.get("approved"):
            return "create_storyboard"
        if state.get("review_note"):
            return "create_story"
        return "review_story"

    def route_after_storyboard_review(state: AgentState) -> str:
        if state.get("storyboard_approved"):
            return "create_visual_storyboard"
        if state.get("storyboard_review_note"):
            return "create_storyboard"
        return "review_storyboard"

    def route_after_visual_review(state: AgentState) -> str:
        if state.get("visual_approved"):
            return "create_scene_videos"
        if state.get("visual_feedback"):
            return "create_visual_storyboard"
        return "review_visual_storyboard"


    graph=StateGraph(AgentState)
    graph.add_node("create_story", story_creator)
    graph.add_node("review_story",review_story)
    graph.add_node("create_storyboard", storyboard_creator)
    graph.add_node("review_storyboard", review_storyboard)
    graph.add_node("create_visual_storyboard", image_creator)
    graph.add_node("review_visual_storyboard", review_visual_storyboard)
    graph.add_node("create_scene_videos", scene_video_creator)
    graph.add_node("create_narration", narration_creator)
    graph.add_node("create_soundfx", soundfx_creator)
    graph.add_node("create_subtitles", subtitle_creator)
    graph.add_node("edit_video", editor)

    # adding edges
    graph.add_edge(START,"create_story")
    graph.add_edge("create_story","review_story")
    graph.add_conditional_edges("review_story",route_after_review)
    graph.add_edge("create_storyboard", "review_storyboard")
    graph.add_conditional_edges("review_storyboard", route_after_storyboard_review)
    graph.add_edge("create_visual_storyboard", "review_visual_storyboard")
    graph.add_conditional_edges("review_visual_storyboard", route_after_visual_review)
    graph.add_edge("create_scene_videos", "create_narration")
    graph.add_edge("create_narration", "create_soundfx")
    graph.add_edge("create_soundfx", "create_subtitles")
    graph.add_edge("create_subtitles", "edit_video")
    graph.add_edge("edit_video", END)
    return graph.compile(checkpointer=InMemorySaver())
