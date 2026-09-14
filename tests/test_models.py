import os
import json
from types import SimpleNamespace

from video_automation import models
from video_automation.models import DEFAULT_HF_MODEL, model_name_for


def test_llm_evaluation_uses_provider_tokens_and_configured_cost(monkeypatch):
    class FakeModel:
        model = "priced-model"

        def invoke(self, _prompt):
            return SimpleNamespace(
                content='{"ok": true}',
                usage_metadata={"input_tokens": 1000, "output_tokens": 250, "total_tokens": 1250},
            )

    monkeypatch.setenv("LLM_INPUT_USD_PER_1M_TOKENS", "2")
    monkeypatch.setenv("LLM_OUTPUT_USD_PER_1M_TOKENS", "8")
    response, evaluation = models.invoke_with_evaluation(
        FakeModel(), "prompt", agent_name="story", purpose="test_call"
    )

    assert response.content == '{"ok": true}'
    assert evaluation.pop("elapsed_seconds") >= 0
    assert evaluation == {
        "agent": "story",
        "purpose": "test_call",
        "model": "priced-model",
        "input_tokens": 1000,
        "output_tokens": 250,
        "total_tokens": 1250,
        "token_source": "provider",
        "cost_usd": 0.004,
    }


def test_llm_evaluation_marks_estimates_and_unknown_cost(monkeypatch):
    monkeypatch.delenv("LLM_INPUT_USD_PER_1M_TOKENS", raising=False)
    monkeypatch.delenv("LLM_OUTPUT_USD_PER_1M_TOKENS", raising=False)

    class FakeModel:
        def invoke(self, _prompt):
            return SimpleNamespace(content="response")

    _, evaluation = models.invoke_with_evaluation(
        FakeModel(), "prompt", agent_name="story", purpose="test_call"
    )
    assert evaluation["token_source"] == "estimated"
    assert evaluation["input_tokens"] > 0
    assert evaluation["output_tokens"] > 0
    assert evaluation["cost_usd"] is None


def test_agent_model_override():
    os.environ.pop("MODEL_PROVIDER", None)
    os.environ["ORCHESTRATOR_MODEL"] = "fast-model"
    try:
        assert model_name_for("orchestrator") == "fast-model"
    finally:
        del os.environ["ORCHESTRATOR_MODEL"]


def test_storyboard_model_override():
    os.environ.pop("MODEL_PROVIDER", None)
    os.environ["STORYBOARD_MODEL"] = "storyboard-model"
    try:
        assert model_name_for("storyboard") == "storyboard-model"
    finally:
        del os.environ["STORYBOARD_MODEL"]


def test_default_model():
    old_hf = os.environ.pop("HF_MODEL", None)
    old_provider = os.environ.pop("MODEL_PROVIDER", None)
    try:
        assert model_name_for("unknown-agent") == DEFAULT_HF_MODEL
    finally:
        if old_hf is not None:
            os.environ["HF_MODEL"] = old_hf
        if old_provider is not None:
            os.environ["MODEL_PROVIDER"] = old_provider


def test_ollama_default_model():
    os.environ["MODEL_PROVIDER"] = "ollama"
    try:
        assert model_name_for("unknown-agent") == os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    finally:
        del os.environ["MODEL_PROVIDER"]


def test_ollama_shot_image_qa_uses_installed_vision_model(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"models": [{"name": "qwen2.5vl:3b"}]}).encode()

    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("QA_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("OLLAMA_QA_MODEL", "qwen2.5vl:3b")
    monkeypatch.delenv("SHOT_IMAGE_QA_MODEL", raising=False)
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    assert models.model_name_for("shot-image-qa") == "qwen2.5vl:3b"
    assert models.validate_model_configuration("shot-image-qa") == "qwen2.5vl:3b"


def test_ollama_preflight_rejects_missing_model(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"models": []}'

    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_QA_MODEL", "missing-vision:latest")
    monkeypatch.delenv("SHOT_IMAGE_QA_MODEL", raising=False)
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    try:
        models.validate_model_configuration("shot-image-qa")
    except RuntimeError as exc:
        assert "not installed" in str(exc)
        return
    raise AssertionError("Ollama preflight should reject an unavailable model")


def test_ollama_client_has_request_timeout(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("MODEL_REQUEST_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("OLLAMA_CONTEXT_TOKENS", "6144")
    monkeypatch.setenv("OLLAMA_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "15m")

    model = models.load_model("story")

    assert model.client_kwargs["timeout"] == 7
    assert model.num_ctx == 6144
    assert model.num_predict == 1024
    assert model.keep_alive == "15m"


def test_llm_evaluation_includes_ollama_timings():
    response = SimpleNamespace(
        content='{"ok": true}',
        usage_metadata={"input_tokens": 2, "output_tokens": 3},
        response_metadata={
            "total_duration": 2_000_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_duration": 600_000_000,
            "eval_duration": 1_300_000_000,
        },
    )

    evaluation = models._evaluation(response, "prompt", "story", "test", "model", elapsed_seconds=2.1)

    assert evaluation["elapsed_seconds"] == 2.1
    assert evaluation["provider_total_seconds"] == 2.0
    assert evaluation["provider_load_seconds"] == 0.1
    assert evaluation["provider_prompt_seconds"] == 0.6
    assert evaluation["provider_generation_seconds"] == 1.3


def test_huggingface_multimodal_message_uses_data_url(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"frame")
    calls = []

    class FakeModel:
        def invoke(self, content):
            calls.append(content)
            return content

    original_loader = models.load_model
    old_provider = os.environ.get("MODEL_PROVIDER")
    os.environ["MODEL_PROVIDER"] = "huggingface"
    models.load_model = lambda _agent: FakeModel()
    try:
        models.invoke_with_images("qa", "judge", [str(image)])
    finally:
        models.load_model = original_loader
        if old_provider is None:
            os.environ.pop("MODEL_PROVIDER", None)
        else:
            os.environ["MODEL_PROVIDER"] = old_provider

    assert calls[0][0] == {"type": "text", "text": "judge"}
    assert calls[0][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_huggingface_chat_preserves_system_messages(monkeypatch):
    calls = []

    class FakeClient:
        def chat_completion(self, **kwargs):
            calls.append(kwargs["messages"])
            usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)
            choice = SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))
            return SimpleNamespace(choices=[choice], usage=usage)

    model = object.__new__(models.HuggingFaceChat)
    model.model = "test-model"
    model.client = FakeClient()
    messages = [
        {"role": "system", "content": "System rules"},
        {"role": "user", "content": "User request"},
    ]
    response = model.invoke(messages)

    assert calls == [messages]
    assert response.usage_metadata == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}


if __name__ == "__main__":
    test_agent_model_override()
    test_storyboard_model_override()
    test_default_model()
    test_ollama_default_model()
