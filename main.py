import json
import uuid

from langgraph.types import Command

from agent_graph import build_story_graph
from schema import AgentState


def get_user_input() -> AgentState:
    topic = input("Video topic: ").strip()
    tone = input ("tone of the video: ").strip()
    duration= input("story duration: ").strip()
    characters = input("Story characters: ").strip()
    language = input("Story language: ").strip()

    output={
            "topic":topic,
            "tone":tone,
            "duration":duration,
            "characters":characters,
            "language":language

            }

    return output

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


def get_storyboard_review_command() -> Command:
    while True:
        action = input("Approve or regenerate storyboard? ").strip().lower()
        if action == "approve":
            return Command(resume={"action": "approve"})
        if action == "regenerate":
            note = input("Storyboard revision note: ").strip()
            return Command(resume={"action": "regenerate", "note": note})
        print("Use approve or regenerate.")


def get_visual_storyboard_review_command() -> Command:
    while True:
        action = input("Approve visuals or regenerate selected scenes? ").strip().lower()
        if action == "approve":
            return Command(resume={"action": "approve"})
        if action == "regenerate":
            feedback = []
            while scene_number := input("Scene number (blank when done): ").strip():
                if not scene_number.isdigit() or int(scene_number) < 1:
                    print("Scene number must be a positive integer.")
                    continue
                note = input(f"Feedback for scene {scene_number}: ").strip()
                if note:
                    feedback.append({"scene_number": int(scene_number), "note": note})
            if feedback:
                return Command(resume={"action": "regenerate", "feedback": feedback})
        print("Use approve, or provide feedback for at least one scene.")

def main():
    thread_id = str(uuid.uuid4())
    graph=build_story_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result=graph.invoke(get_user_input(),config)

    while "__interrupt__"  in result:
        review = result["__interrupt__"][0].value
        if "image_files" in review:
            print("Mood board:", review["mood_board_file"])
            print(json.dumps(list(zip(review["storyboard"], review["image_files"])), indent=2))
            result = graph.invoke(get_visual_storyboard_review_command(), config)
        elif "storyboard" in review:
            print(json.dumps(review["storyboard"], indent=2))
            result = graph.invoke(get_storyboard_review_command(), config)
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
