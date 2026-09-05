import os

from models import DEFAULT_HF_MODEL, model_name_for


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


if __name__ == "__main__":
    test_agent_model_override()
    test_storyboard_model_override()
    test_default_model()
    test_ollama_default_model()
