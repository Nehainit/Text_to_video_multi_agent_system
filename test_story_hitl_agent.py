from langgraph.types import Command

import agent_graph


def fake_story_creator(state):
    note = state.get("review_note", "")
    return {"story": f"Draft story. {note}".strip()}


def fake_storyboard_creator(state):
    note = state.get("storyboard_review_note", "")
    return {"storyboard": [{"scene_number": 1, "start_time": "00:00", "end_time": "00:10", "visuals": f'{state["story"]} {note}'.strip(), "camera": "wide", "narration": state["story"]}]}


def fake_image_creator(state):
    feedback = state.get("visual_feedback", [])
    suffix = "-revised" if feedback else ""
    return {
        "mood_board_file": "outputs/test/mood_board.png",
        "image_files": [f"outputs/test/images/scene_01{suffix}.png"],
        "visual_feedback": [],
        "last_visual_feedback": feedback,
    }


def fake_scene_video_creator(state):
    return {"scene_video_files": []}


def fake_narration_creator(state):
    return {"narration_file": "outputs/test/audio/narration.mp3"}


def fake_soundfx_creator(state):
    return {"sfx_files": ["outputs/test/audio/sfx_scene_01.mp3"]}


def fake_subtitle_creator(state):
    return {"subtitles": [{"scene_number": 1, "text": state["story"]}], "subtitle_file": "outputs/test/subtitles/story.srt"}


def fake_video_creator(state):
    return {"mixed_audio_file": "outputs/test/video/mixed_audio.mp3", "video_file": "outputs/test/video/final_reel.mp4"}


def build_fake_graph():
    return agent_graph.build_story_graph(
        fake_story_creator,
        fake_storyboard_creator,
        fake_image_creator,
        fake_narration_creator,
        fake_soundfx_creator,
        fake_subtitle_creator,
        fake_video_creator,
        scene_video_creator=fake_scene_video_creator,
    )


def test_interrupt_then_approve():
    graph = build_fake_graph()
    config = {"configurable": {"thread_id": "hitl-approve"}}
    state = {
        "topic": "Raja and Rani are trapped in a mystical forest by a devil",
        "tone": "mystical",
        "duration": "1 minute",
        "language": "English",
        "characters": ["Raja", "Rani", "Devil"],
    }

    first = graph.invoke(state, config)
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["actions"] == ["approve", "edit", "regenerate"]

    storyboard_review = graph.invoke(Command(resume={"action": "approve"}), config)
    assert "__interrupt__" in storyboard_review
    assert storyboard_review["__interrupt__"][0].value["actions"] == ["approve", "regenerate"]

    visual_review = graph.invoke(Command(resume={"action": "approve"}), config)
    assert "__interrupt__" in visual_review
    assert visual_review["__interrupt__"][0].value["actions"] == ["approve", "regenerate"]

    final = graph.invoke(Command(resume={"action": "approve"}), config)
    assert final["approved"] is True
    assert final["story"] == "Draft story."
    assert final["storyboard"][0]["scene_number"] == 1
    assert final["image_files"] == ["outputs/test/images/scene_01.png"]
    assert final["narration_file"] == "outputs/test/audio/narration.mp3"
    assert final["sfx_files"] == ["outputs/test/audio/sfx_scene_01.mp3"]
    assert final["subtitle_file"] == "outputs/test/subtitles/story.srt"
    assert final["video_file"] == "outputs/test/video/final_reel.mp4"


def test_edit_resume_replaces_story():
    graph = build_fake_graph()
    config = {"configurable": {"thread_id": "hitl-edit"}}
    graph.invoke(
        {
            "topic": "x",
            "tone": "y",
            "duration": "1 minute",
            "language": "English",
            "characters": ["Raja"],
        },
        config,
    )

    edited = graph.invoke(Command(resume={"action": "edit", "story": "Edited story."}), config)
    assert "__interrupt__" in edited
    assert edited["approved"] is False
    assert edited["story"] == "Edited story."

    storyboard_review = graph.invoke(Command(resume={"action": "approve"}), config)
    assert "__interrupt__" in storyboard_review

    visual_review = graph.invoke(Command(resume={"action": "approve"}), config)
    assert "__interrupt__" in visual_review

    final = graph.invoke(Command(resume={"action": "approve"}), config)
    assert final["approved"] is True
    assert final["story"] == "Edited story."


def test_regenerate_routes_back_to_story():
    graph = build_fake_graph()
    config = {"configurable": {"thread_id": "hitl-regenerate"}}
    graph.invoke(
        {
            "topic": "x",
            "tone": "y",
            "duration": "1 minute",
            "language": "English",
            "characters": ["Raja"],
        },
        config,
    )

    second = graph.invoke(Command(resume={"action": "regenerate", "note": "make it warmer"}), config)
    assert "__interrupt__" in second
    assert second["story"] == "Draft story. make it warmer"


def test_regenerate_routes_back_to_storyboard():
    graph = build_fake_graph()
    config = {"configurable": {"thread_id": "hitl-storyboard-regenerate"}}
    graph.invoke(
        {
            "topic": "x",
            "tone": "y",
            "duration": "1 minute",
            "language": "English",
            "characters": ["Raja"],
        },
        config,
    )
    graph.invoke(Command(resume={"action": "approve"}), config)

    revised = graph.invoke(
        Command(resume={"action": "regenerate", "note": "use closer camera shots"}),
        config,
    )
    assert "__interrupt__" in revised
    assert revised["storyboard"][0]["visuals"].endswith("use closer camera shots")
    assert revised["__interrupt__"][0].value["feedback"] == "use closer camera shots"


def test_regenerates_selected_visual_scenes():
    graph = build_fake_graph()
    config = {"configurable": {"thread_id": "hitl-visual-regenerate"}}
    graph.invoke(
        {
            "topic": "x",
            "tone": "y",
            "duration": "1 minute",
            "language": "English",
            "characters": ["Raja"],
        },
        config,
    )
    graph.invoke(Command(resume={"action": "approve"}), config)
    visual_review = graph.invoke(Command(resume={"action": "approve"}), config)
    assert "image_files" in visual_review["__interrupt__"][0].value

    revised = graph.invoke(
        Command(resume={"action": "regenerate", "feedback": [{"scene_number": 1, "note": "warmer light"}]}),
        config,
    )
    assert revised["image_files"] == ["outputs/test/images/scene_01-revised.png"]
    assert revised["__interrupt__"][0].value["feedback"] == [{"scene_number": 1, "note": "warmer light"}]
    assert revised["visual_feedback_history"][0]["revision"] == 1


if __name__ == "__main__":
    test_interrupt_then_approve()
    test_edit_resume_replaces_story()
    test_regenerate_routes_back_to_story()
    test_regenerate_routes_back_to_storyboard()
    test_regenerates_selected_visual_scenes()
