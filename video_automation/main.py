import json
import uuid

from langgraph.types import Command

from video_automation.agent_graph import build_story_graph
from video_automation.schema import AgentState


def get_user_input() -> AgentState:
    topic = input("Describe the story, characters, mood, and visual style: ").strip()
    duration = input("Duration in seconds: ").strip()
    language = input("Story language: ").strip()
    aspect_ratio = input("Aspect ratio (16:9, 9:16, or 1:1): ").strip() or "9:16"
    video_quality = input("Video quality (standard or high): ").strip().lower() or "standard"

    return {
        "topic": topic,
        "tone": "Infer the tone and visual style from the user's prompt.",
        "duration": f"{duration} seconds",
        "characters": "",
        "language": language,
        "aspect_ratio": aspect_ratio,
        "video_quality": video_quality,
    }

def get_review_command() -> Command:
    while True:
        action = input("Approve, edit, or regenerate? ").strip().lower()
        if action == "approve":
            return Command(resume={"action": "approve"})
        if action == "edit":
            edited_story = input("Edited story: ").strip()
            return Command(resume={"action":"edit","story":edited_story})
        if action == "regenerate":
            note=input("revision note : ").strip()
            return Command(resume={"action":"regenerate","note":note})
        print("Use approve, edit, or regenerate.")


def get_reference_review_command() -> Command:
    while True:
        action = input("Approve or regenerate the character and mood boards? ").strip().lower()
        if action == "approve":
            return Command(resume={"action": "approve"})
        if action == "regenerate":
            note = input("Reference package revision note: ").strip()
            return Command(resume={"action": "regenerate", "note": note})
        print("Use approve or regenerate.")


def get_shot_review_command(label: str) -> Command:
    while True:
        action = input(f"Approve {label} or regenerate selected shots? ").strip().lower()
        if action == "approve":
            return Command(resume={"action": "approve"})
        if action == "regenerate":
            feedback = []
            while shot_id := input("Shot ID such as shot-002 (blank when done): ").strip():
                note = input(f"Feedback for {shot_id}: ").strip()
                if note:
                    feedback.append({"shot_id": shot_id, "note": note})
            if feedback:
                return Command(resume={"action": "regenerate", "feedback": feedback})
        print("Use approve, or provide feedback for at least one shot.")

def main():
    thread_id = str(uuid.uuid4())
    graph=build_story_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = get_user_input()
    initial_state["thread_id"] = thread_id
    result=graph.invoke(initial_state,config)

    while "__interrupt__"  in result:
        review = result["__interrupt__"][0].value
        if "image_files" in review:
            print(json.dumps(list(zip(review["storyboard"], review["image_files"])), indent=2))
            result = graph.invoke(get_shot_review_command("visual storyboard"), config)
        elif "character_board_file" in review:
            print("Character board:", review["character_board_file"])
            print("Mood board:", review["mood_board_file"])
            result = graph.invoke(get_reference_review_command(), config)
        elif "director_plan" in review:
            print(json.dumps(review["director_plan"], indent=2))
            result = graph.invoke(get_shot_review_command("Director plan"), config)
        else:
            print(review["story"])
            result = graph.invoke(get_review_command(), config)

    print(result["story"])
    print(json.dumps(result.get("storyboard", []), indent=2))
    print(json.dumps(result.get("image_files", []), indent=2))
    print(result.get("mood_board_file", ""))
    print(result.get("narration_file", ""))
    print(json.dumps(result.get("sfx_files", []), indent=2))
    print(result.get("subtitle_file", ""))
    print(result.get("video_file", ""))

if __name__ == "__main__":
    main()
