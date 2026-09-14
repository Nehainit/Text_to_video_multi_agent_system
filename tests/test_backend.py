from types import SimpleNamespace

from video_automation import backend
import pytest


@pytest.fixture(autouse=True)
def _configured_vision_model(monkeypatch):
    monkeypatch.setattr(backend, "validate_model_configuration", lambda _agent: "vision-model")


SHOT = {"shot_id": "shot-001", "scene_id": "scene-001", "scene_number": 1, "visuals": "Opening resolves."}


class FakeGraph:
    def __init__(self):
        self.next_node = "review_story"
        self.commands = []
        self.initial_state = None

    def get_state(self, _config):
        return SimpleNamespace(
            next=(self.next_node,) if self.next_node else (),
            values={"storyboard": [SHOT], "director_plan": [SHOT]},
        )

    def invoke(self, value, _config):
        if isinstance(value, dict):
            self.initial_state = value
            return {"__interrupt__": [SimpleNamespace(value={"story": "Draft story.", "actions": ["approve", "edit", "regenerate"]})]}
        self.commands.append(value)
        if self.next_node == "review_story":
            self.next_node = "review_visual_storyboard"
            return {
                "story": "Draft story.",
                "production_bible": {"theme": "homecoming"},
                "character_board_file": str(backend.ROOT / "outputs/test/character.png"),
                "mood_board_file": str(backend.ROOT / "outputs/test/mood.png"),
                "storyboard": [SHOT],
                "image_files": [str(backend.ROOT / "outputs/test/images/shot-001.png")],
                "__interrupt__": [
                    SimpleNamespace(value={"storyboard": [SHOT], "image_files": [str(backend.ROOT / "outputs/test/images/shot-001.png")], "actions": ["approve", "regenerate"]})
                ],
            }
        self.next_node = ""
        return {
            "story": "Draft story.",
            "storyboard": [SHOT],
            "warnings": ["local fallback"],
            "video_file": str(backend.ROOT / "outputs/test.mp4"),
        }


def test_web_request_auto_advances_review_checkpoints(monkeypatch):
    monkeypatch.setenv("QA_MODEL", "vision-model")
    original_graph = backend.graph
    fake_graph = FakeGraph()
    backend.graph = fake_graph
    try:
        response = backend.create_video(backend.VideoRequest(topic="A funny story about Mira", duration=47, language="Hindi"))
        assert response["status"] == "complete"
        assert fake_graph.initial_state["duration"] == "47 seconds"
        assert fake_graph.initial_state["thread_id"] == response["thread_id"]
        assert fake_graph.initial_state["quality_mode"] == "standard"
        assert fake_graph.initial_state["aspect_ratio"] == "9:16"
        assert fake_graph.initial_state["video_quality"] == "standard"
        assert response["warnings"] == ["local fallback"]
        assert response["video_url"] == "/output/outputs/test.mp4"
        assert [command.resume for command in fake_graph.commands] == [{"action": "approve", "note": ""}] * 2
    finally:
        backend.graph = original_graph


def test_web_request_stops_repeating_the_same_review(monkeypatch):
    class LoopingReviewGraph:
        def __init__(self):
            self.commands = []

        def invoke(self, value, _config):
            if not isinstance(value, dict):
                self.commands.append(value)
            return {
                "story": "Draft story.",
                "shot_plan_revision_reason": "A shot still contains sequential actions.",
                "__interrupt__": [SimpleNamespace(value={
                    "stage": "shot_plan_review",
                    "revision_reason": "A shot still contains sequential actions.",
                    "issues": ["shot-001 needs one visible action."],
                    "actions": ["retry", "revise_visuals"],
                })],
            }

    monkeypatch.setenv("QA_MODEL", "vision-model")
    original_graph = backend.graph
    fake_graph = LoopingReviewGraph()
    backend.graph = fake_graph
    try:
        response = backend.create_video(backend.VideoRequest(topic="A story"))
    finally:
        backend.graph = original_graph

    assert response["status"] == "shot_plan_review"
    assert [command.resume["action"] for command in fake_graph.commands] == ["retry", "retry", "revise_visuals"]


def test_web_request_stops_repeating_story_review(monkeypatch):
    class LoopingStoryGraph:
        def __init__(self):
            self.commands = []

        def invoke(self, value, _config):
            if not isinstance(value, dict):
                self.commands.append(value)
            return {
                "story": "Draft story.",
                "__interrupt__": [SimpleNamespace(value={
                    "stage": "story_review",
                    "story": "Draft story.",
                    "actions": ["approve", "edit", "regenerate"],
                })],
            }

    monkeypatch.setenv("QA_MODEL", "vision-model")
    original_graph = backend.graph
    fake_graph = LoopingStoryGraph()
    backend.graph = fake_graph
    try:
        response = backend.create_video(backend.VideoRequest(topic="A story"))
    finally:
        backend.graph = original_graph

    assert response["status"] == "story_review"
    assert len(fake_graph.commands) == backend.MAX_AUTO_ADVANCE_STEPS


def test_web_request_recovers_scene_plan_upstream_and_continues(monkeypatch):
    class RecoveringGraph:
        def __init__(self):
            self.commands = []

        def invoke(self, value, _config):
            if isinstance(value, dict):
                return self._paused()
            self.commands.append(value)
            if value.resume["action"] == "revise_narration":
                return {
                    "story": "Draft story.",
                    "storyboard": [SHOT],
                    "warnings": [],
                    "video_file": str(backend.ROOT / "outputs/recovered.mp4"),
                }
            return self._paused()

        @staticmethod
        def _paused():
            return {
                "story": "Draft story.",
                "scene_plan_revision_reason": "Narration is too short for three scenes.",
                "__interrupt__": [SimpleNamespace(value={
                    "stage": "scene_plan_review",
                    "revision_reason": "Narration is too short for three scenes.",
                    "issues": ["Three scenes need more narration."],
                    "actions": ["retry", "revise_narration"],
                })],
            }

    monkeypatch.setenv("QA_MODEL", "vision-model")
    original_graph = backend.graph
    fake_graph = RecoveringGraph()
    backend.graph = fake_graph
    try:
        response = backend.create_video(backend.VideoRequest(topic="A three-scene story"))
    finally:
        backend.graph = original_graph

    assert response["status"] == "complete"
    assert [command.resume["action"] for command in fake_graph.commands] == ["retry", "retry", "revise_narration"]


def test_rejects_duration_outside_supported_range():
    try:
        backend.VideoRequest(topic="A story", duration=4)
    except ValueError:
        return
    raise AssertionError("duration shorter than five seconds should be rejected")


def test_refined_quality_mode_is_accepted():
    assert backend.VideoRequest(topic="A story", quality_mode="refine").quality_mode == "refine"


def test_aspect_ratio_and_output_quality_are_accepted():
    request = backend.VideoRequest(topic="A story", aspect_ratio="16:9", video_quality="high")
    assert request.aspect_ratio == "16:9"
    assert request.video_quality == "high"


def test_unsafe_request_returns_bad_request(monkeypatch):
    monkeypatch.setenv("QA_MODEL", "vision-model")
    class UnsafeGraph:
        def invoke(self, _value, _config):
            raise backend.UnsafeContentError(["self_harm"])

    original_graph = backend.graph
    backend.graph = UnsafeGraph()
    try:
        try:
            backend.create_video(backend.VideoRequest(topic="Unsafe request"))
        except Exception as exc:
            assert exc.status_code == 400
            assert exc.detail["categories"] == ["self_harm"]
        else:
            raise AssertionError("Unsafe request was not returned as a client error")
    finally:
        backend.graph = original_graph


def test_scene_plan_failure_is_returned_for_human_review():
    response = backend._graph_response(
        {
            "story": "Approved story.",
            "scene_plan_revision_reason": "Scene timing could not be validated.",
            "scene_plan_issues": ["scene-001 exceeds the narration timing."],
            "__interrupt__": [SimpleNamespace(value={
                "stage": "scene_plan_review",
                "revision_reason": "Scene timing could not be validated.",
                "issues": ["scene-001 exceeds the narration timing."],
                "actions": ["retry", "revise_narration"],
            })],
        },
        "thread",
    )

    assert response["status"] == "scene_plan_review"
    assert response["actions"] == ["retry", "revise_narration"]


def test_visual_plan_failure_is_returned_for_human_review():
    response = backend._graph_response(
        {
            "story": "Approved story.",
            "visual_plan_revision_reason": "Visual coverage could not be validated.",
            "visual_plan_issues": ["scene-001 has no visible action."],
            "__interrupt__": [SimpleNamespace(value={
                "stage": "visual_plan_review",
                "revision_reason": "Visual coverage could not be validated.",
                "issues": ["scene-001 has no visible action."],
                "actions": ["retry", "revise_scenes"],
            })],
        },
        "thread",
    )

    assert response["status"] == "visual_plan_review"
    assert response["actions"] == ["retry", "revise_scenes"]


def test_shot_plan_failure_is_returned_for_human_review():
    response = backend._graph_response(
        {
            "story": "Approved story.",
            "shot_plan_revision_reason": "A visual beat cannot form a clear keyframe.",
            "shot_plan_issues": ["vb-001 contains sequential actions."],
            "__interrupt__": [SimpleNamespace(value={
                "stage": "shot_plan_review",
                "revision_reason": "A visual beat cannot form a clear keyframe.",
                "issues": ["vb-001 contains sequential actions."],
                "actions": ["retry", "revise_visuals"],
            })],
        },
        "thread",
    )

    assert response["status"] == "shot_plan_review"
    assert response["actions"] == ["retry", "revise_visuals"]


def test_video_request_requires_vision_model(monkeypatch):
    monkeypatch.setattr(
        backend,
        "validate_model_configuration",
        lambda _agent: (_ for _ in ()).throw(RuntimeError("Vision model is unavailable.")),
    )
    try:
        backend.create_video(backend.VideoRequest(topic="A story"))
    except Exception as exc:
        assert exc.status_code == 400
        return
    raise AssertionError("refined mode should require QA_MODEL")


def test_output_endpoint_serves_only_generated_artifacts(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    video = outputs / "video.mp4"
    video.write_bytes(b"video")
    secret = tmp_path / ".env"
    secret.write_text("SECRET=value")
    monkeypatch.setattr(backend, "ROOT", tmp_path)

    assert backend.output_file("outputs/video.mp4").path == video
    with pytest.raises(backend.HTTPException) as error:
        backend.output_file(".env")
    assert error.value.status_code == 404


def test_story_regeneration_requires_feedback():
    try:
        backend.submit_story_review(backend.StoryReviewRequest(thread_id="thread", action="regenerate", note=" "))
    except Exception as exc:
        assert exc.status_code == 400
        return
    raise AssertionError("blank story feedback should be rejected")


def test_approve_retries_failed_downstream_node():
    class FailedGraph:
        def __init__(self):
            self.inputs = []

        def get_state(self, _config):
            return SimpleNamespace(next=("create_visual_beats",), values={"approved": True})

        def invoke(self, value, _config):
            self.inputs.append(value)
            return {"story": "Approved story.", "storyboard": [], "warnings": [], "video_file": None}

    original_graph = backend.graph
    failed_graph = FailedGraph()
    backend.graph = failed_graph
    try:
        response = backend.submit_story_review(backend.StoryReviewRequest(thread_id="thread", action="approve"))
    finally:
        backend.graph = original_graph

    assert response["status"] == "complete"
    assert failed_graph.inputs == [None]


if __name__ == "__main__":
    test_web_story_then_visual_storyboard_flow()
    test_rejects_duration_outside_supported_range()
    test_story_regeneration_requires_feedback()
