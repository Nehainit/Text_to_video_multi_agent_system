import os
from types import SimpleNamespace

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = lambda *args, **kwargs: None


load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

AGENT_MODEL_ENV = {
    "orchestrator": "ORCHESTRATOR_MODEL",
    "story": "STORY_MODEL",
    "storyboard": "STORYBOARD_MODEL",
    "scene": "SCENE_MODEL",
    "qa": "QA_MODEL",
}
DEFAULT_HF_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_HF_PROVIDER = "nscale"


def _provider() -> str:
    return os.getenv("MODEL_PROVIDER", "huggingface").lower()


def model_name_for(agent_name: str) -> str:
    env_name = AGENT_MODEL_ENV.get(agent_name)
    if _provider() == "ollama":
        return (os.getenv(env_name) if env_name else None) or os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    return (os.getenv(env_name) if env_name else None) or os.getenv("HF_MODEL", DEFAULT_HF_MODEL)


class HuggingFaceChat:
    def __init__(self, model: str):
        from huggingface_hub import InferenceClient

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token or not token.startswith("hf_"):
            raise RuntimeError("Set a valid Hugging Face token in HF_TOKEN, or set MODEL_PROVIDER=ollama.")
        self.client = InferenceClient(
            model=model,
            provider=os.getenv("HF_PROVIDER") or DEFAULT_HF_PROVIDER,
            token=token,
            timeout=180,
        )

    def invoke(self, prompt: str):
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(os.getenv("HF_MAX_TOKENS", "2048")),
                temperature=float(os.getenv("HF_TEMPERATURE", "0.3")),
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "401" in str(exc) or "Unauthorized" in str(exc):
                raise RuntimeError("Hugging Face rejected HF_TOKEN. Replace it with a valid token, or set MODEL_PROVIDER=ollama.") from exc
            if "model_not_supported" in str(exc) or "not supported by any provider" in str(exc):
                raise RuntimeError(
                    f"Hugging Face cannot run {model!r} with your enabled provider. "
                    "Use HF_MODEL=Qwen/Qwen3-4B-Thinking-2507 with HF_PROVIDER=nscale, "
                    "or enable the model provider in Hugging Face."
                ) from exc
            raise
        return SimpleNamespace(content=response.choices[0].message.content)


def load_model(agent_name: str):
    if _provider() == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_name_for(agent_name), temperature=0.3, format="json")
    return HuggingFaceChat(model_name_for(agent_name))
