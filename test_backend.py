from types import SimpleNamespace

import backend


class FakeGraph:
    def __init__(self):
        self.next_node = "review_story"
        self.commands = []
        self.storyboard = [{"scene_number": 1, "visuals": "Opening scene"}]
        self.image_files = [str(backend.ROOT / "outputs/test/images/scene_01.png")]
        self.mood_board_file = str(backend.ROOT / "outputs/test/mood_board.png")

    def get_state(self, _config):
        return SimpleNamespace(next=(self.next_node,) if self.next_node else (), values={"storyboard": self.storyboard})

    def invoke(self, value, _config):
        if isinstance(value, dict):
            return {
                "__interrupt__": [
                    SimpleNamespace(
                        value={"story": "Draft story.", "actions": ["approve", "edit", "regenerate"]}
                    )
                ]
            }
        self.commands.append(value)
        if self.next_node == "review_story":
            self.next_node = "review_storyboard"
            return {
                "story": "Draft story.",
                "storyboard": self.storyboard,
                "__interrupt__": [
                    SimpleNamespace(value={"storyboard": self.storyboard, "actions": ["approve", "regenerate"]})
                ],
            }
        if self.next_node == "review_storyboard":
            self.next_node = "review_visual_storyboard"
            return {
                "story": "Draft story.",
                "storyboard": self.storyboard,
                "mood_board_file": self.mood_board_file,
                "image_files": self.image_files,
                "__interrupt__": [
                    SimpleNamespace(
                        value={
                            "storyboard": self.storyboard,
                            "mood_board_file": self.mood_board_file,
                            "image_files": self.image_files,
                            "actions": ["approve", "regenerate"],
                            "feedback": [],
                        }
                    )
                ],
            }
        self.next_node = ""
        return {
            "story": "Draft story.",
            "storyboard": self.storyboard,
            "video_file": str(backend.ROOT / "outputs/test.mp4"),
        }


def test_web_story_review_flow():
    original_graph = backend.graph
    fake_graph = FakeGraph()
    backend.graph = fake_graph
    try:
        review = backend.create_video(backend.VideoRequest(topic="A short story"))
        assert review["status"] == "review"
        assert review["story"] == "Draft story."

        storyboard_review = backend.submit_story_review(
            backend.StoryReviewRequest(thread_id=review["thread_id"], action="approve")
        )
        assert storyboard_review["status"] == "storyboard_review"
        assert storyboard_review["artifacts"]["storyboard"][0]["scene_number"] == 1

        complete = backend.submit_storyboard_review(
            backend.StoryboardReviewRequest(thread_id=review["thread_id"], action="approve")
        )
        assert complete["status"] == "visual_storyboard_review"
        assert complete["image_urls"] == ["/output/outputs/test/images/scene_01.png"]

        complete = backend.submit_visual_storyboard_review(
            backend.VisualStoryboardReviewRequest(thread_id=review["thread_id"], action="approve")
        )
        assert [command.resume for command in fake_graph.commands] == [
            {"action": "approve"},
            {"action": "approve"},
            {"action": "approve"},
        ]
        assert complete["status"] == "complete"
        assert complete["video_url"] == "/output/outputs/test.mp4"
        assert complete["artifacts"]["story"] == "Draft story."
        assert complete["artifacts"]["storyboard"] == fake_graph.storyboard
    finally:
        backend.graph = original_graph


if __name__ == "__main__":
    test_web_story_review_flow()
