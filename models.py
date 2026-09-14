import base64
import json
import math
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = lambda *args, **kwargs: None


load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

AGENT_MODEL_ENV = {
    "orchestrator": "ORCHESTRATOR_MODEL",
    "safety": "SAFETY_MODEL",
    "story": "STORY_MODEL",
    "narration": "NARRATION_MODEL",
    "director": "DIRECTOR_MODEL",
    "storyboard": "STORYBOARD_MODEL",
    "scene": "SCENE_MODEL",
    "visual-beat": "VISUAL_BEAT_MODEL",
    "shot-planner": "SHOT_PLANNER_MODEL",
    "image-prompt-builder": "IMAGE_PROMPT_BUILDER_MODEL",
    "shot-image-qa": "SHOT_IMAGE_QA_MODEL",
    "combined-video-judge": "OLLAMA_QA_MODEL",
    "motion-planner": "MOTION_PLANNER_MODEL",
    "soundfx": "SOUNDFX_MODEL",
}
DEFAULT_HF_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_HF_PROVIDER = "nscale"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def _request_timeout_seconds() -> float:
    return float(os.getenv("MODEL_REQUEST_TIMEOUT_SECONDS", "180"))


def _provider() -> str:
    return os.getenv("MODEL_PROVIDER", "huggingface").lower()


def model_name_for(agent_name: str) -> str:
    env_name = AGENT_MODEL_ENV.get(agent_name)
    configured = os.getenv(env_name) if env_name else None
    provider = _provider()
    if provider == "ollama":
        if agent_name == "shot-image-qa":
            return configured or os.getenv("OLLAMA_QA_MODEL", "qwen2.5vl:3b")
        return configured or os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    if env_name == "OLLAMA_QA_MODEL":
        configured = None
    configured = configured or (os.getenv("QA_MODEL") if agent_name == "shot-image-qa" else None)
    if provider in {"gemini", "google"}:
        gemini_override = os.getenv(f"GEMINI_{agent_name.replace('-', '_').upper()}_MODEL")
        return configured or gemini_override or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if provider == "groq":
        groq_override = os.getenv(f"GROQ_{agent_name.replace('-', '_').upper()}_MODEL")
        return configured or groq_override or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return configured or os.getenv("HF_MODEL", DEFAULT_HF_MODEL)


def validate_model_configuration(agent_name: str) -> str:
    provider = _provider()
    if agent_name == "shot-image-qa" and provider != "ollama" and not (
        os.getenv("SHOT_IMAGE_QA_MODEL") or os.getenv("QA_MODEL")
    ):
        raise RuntimeError("Set SHOT_IMAGE_QA_MODEL or QA_MODEL to a vision-capable model.")
    model = model_name_for(agent_name)
    if provider != "ollama":
        return model
    base_url = (os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=min(5, _request_timeout_seconds())) as response:
            available = {
                value
                for item in json.loads(response.read().decode("utf-8")).get("models", [])
                for value in (item.get("name"), item.get("model")) if value
            }
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Cannot reach Ollama at {base_url} to validate {model!r}.") from exc
    aliases = {model, f"{model}:latest"} if ":" not in model else {model}
    if not aliases & available:
        raise RuntimeError(f"Ollama model {model!r} is not installed.")
    return model


class HuggingFaceChat:
    def __init__(self, model: str):
        from huggingface_hub import InferenceClient

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token or not token.startswith("hf_"):
            raise RuntimeError("Set a valid Hugging Face token in HF_TOKEN, or set MODEL_PROVIDER=ollama.")
        self.model = model
        self.client = InferenceClient(
            model=model,
            provider=os.getenv("HF_PROVIDER") or DEFAULT_HF_PROVIDER,
            token=token,
            timeout=_request_timeout_seconds(),
        )

    def invoke(self, prompt: str | list[dict]):
        try:
            messages = prompt if isinstance(prompt, list) and all(
                isinstance(message, dict) and {"role", "content"} <= message.keys() for message in prompt
            ) else [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=int(os.getenv("HF_MAX_TOKENS", "2048")),
                temperature=float(os.getenv("HF_TEMPERATURE", "0.3")),
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "401" in str(exc) or "Unauthorized" in str(exc):
                raise RuntimeError("Hugging Face rejected HF_TOKEN. Replace it with a valid token, or set MODEL_PROVIDER=ollama.") from exc
            if "model_not_supported" in str(exc) or "not supported by any provider" in str(exc):
                raise RuntimeError(
                    f"Hugging Face cannot run {self.model!r} with your enabled provider. "
                    "Use HF_MODEL=Qwen/Qwen3-4B-Thinking-2507 with HF_PROVIDER=nscale, "
                    "or enable the model provider in Hugging Face."
                ) from exc
            raise
        usage = getattr(response, "usage", None)
        return SimpleNamespace(
            content=response.choices[0].message.content,
            usage_metadata={
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            } if usage else None,
        )


class GeminiChat:
    """Small adapter matching the existing model.invoke() contract."""

    def __init__(self, model: str):
        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install google-genai to use MODEL_PROVIDER=gemini.") from exc
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY when MODEL_PROVIDER=gemini.")
        self.model = model
        self._types = types
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(_request_timeout_seconds() * 1000)),
        )

    def invoke(self, prompt):
        system = []
        parts = []
        if isinstance(prompt, str):
            parts.append(prompt)
        elif isinstance(prompt, list):
            for item in prompt:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    data_url = item.get("image_url", {}).get("url", "")
                    header, encoded = data_url.split(",", 1)
                    mime_type = header.split(";", 1)[0].removeprefix("data:")
                    parts.append(self._types.Part.from_bytes(
                        data=base64.b64decode(encoded), mime_type=mime_type
                    ))
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, dict) and {"role", "content"} <= item.keys():
                    target = system if item["role"] == "system" else parts
                    target.append(f"{item['role'].upper()}:\n{item['content']}")
                else:
                    parts.append(str(item))
        else:
            parts.append(str(prompt))
        response = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            config=self._types.GenerateContentConfig(
                system_instruction="\n\n".join(system) or None,
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
                response_mime_type="application/json",
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        return SimpleNamespace(
            content=getattr(response, "text", ""),
            usage_metadata={
                "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            } if usage else None,
        )


class GroqChat:
    """Minimal Groq adapter using its OpenAI-compatible HTTP endpoint."""

    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_api_key") or os.getenv("groq_api_key")
        if not self.api_key:
            raise RuntimeError("Set GROQ_API_KEY when MODEL_PROVIDER=groq.")

    def invoke(self, prompt):
        messages = prompt if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt) else [
            {"role": "user", "content": prompt}
        ]
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.3")),
            "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", "2048")),
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "magnifc-video-automation-agent/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_request_timeout_seconds()) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 403 and "1010" in detail:
                raise RuntimeError(
                    "Groq rejected the request at its edge (HTTP 403 code 1010). "
                    "Check network access or use a non-blocked User-Agent."
                ) from exc
            if exc.code == 403:
                raise RuntimeError(
                    f"Groq denied model {self.model!r} (HTTP 403). Enable this model in the Groq "
                    "organization/project model permissions, then retry."
                ) from exc
            raise RuntimeError(f"Groq request failed: HTTP {exc.code} {detail}") from exc
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return SimpleNamespace(
            content=choice["message"]["content"],
            usage_metadata={
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            } if usage else None,
        )

def load_model(agent_name: str):
    if _provider() == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name_for(agent_name),
            temperature=0.3,
            format="json",
            num_ctx=int(os.getenv("OLLAMA_CONTEXT_TOKENS", "8192")),
            num_predict=int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "2048")),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
            client_kwargs={"timeout": _request_timeout_seconds()},
        )
    if _provider() in {"gemini", "google"}:
        return GeminiChat(model_name_for(agent_name))
    if _provider() == "groq":
        return GroqChat(model_name_for(agent_name))
    if _provider() == "bedrock":
        raise RuntimeError("MODEL_PROVIDER=bedrock is reserved for a future Bedrock adapter.")
    return HuggingFaceChat(model_name_for(agent_name))


def _token_usage(response, prompt) -> tuple[int, int, str]:
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    usage = usage or metadata.get("token_usage") or metadata.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if input_tokens is not None and output_tokens is not None:
        return int(input_tokens), int(output_tokens), "provider"
    prompt_text = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False, default=str)
    output_text = str(getattr(response, "content", ""))
    return max(1, math.ceil(len(prompt_text) / 4)), max(1, math.ceil(len(output_text) / 4)), "estimated"


def _price(name: str) -> float | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    price = float(value)
    if price < 0:
        raise ValueError(f"{name} cannot be negative.")
    return price


def invoke_with_evaluation(model, prompt, *, agent_name: str, purpose: str):
    """Invoke one LLM and return its response plus token/cost evaluation."""
    started = time.monotonic()
    response = model.invoke(prompt)
    return response, _evaluation(
        response, prompt, agent_name, purpose,
        getattr(model, "model", model_name_for(agent_name)),
        elapsed_seconds=time.monotonic() - started,
    )


def _evaluation(response, prompt, agent_name: str, purpose: str, model: str, *, elapsed_seconds: float | None = None) -> dict:
    input_tokens, output_tokens, source = _token_usage(response, prompt)
    input_rate = _price("LLM_INPUT_USD_PER_1M_TOKENS")
    output_rate = _price("LLM_OUTPUT_USD_PER_1M_TOKENS")
    cost = None if input_rate is None or output_rate is None else round(
        (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000, 10
    )
    result = {
        "agent": agent_name,
        "purpose": purpose,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "token_source": source,
        "cost_usd": cost,
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
    }
    metadata = getattr(response, "response_metadata", None) or {}
    for source_key, target_key in (
        ("total_duration", "provider_total_seconds"),
        ("load_duration", "provider_load_seconds"),
        ("prompt_eval_duration", "provider_prompt_seconds"),
        ("eval_duration", "provider_generation_seconds"),
    ):
        if metadata.get(source_key) is not None:
            result[target_key] = round(float(metadata[source_key]) / 1_000_000_000, 3)
    return result


def invoke_with_images(agent_name: str, prompt: str, image_files: list[str]):
    images = []
    for file_path in image_files:
        mime_type = mimetypes.guess_type(file_path)[0] or "image/jpeg"
        encoded = base64.b64encode(Path(file_path).read_bytes()).decode("ascii")
        images.append(f"data:{mime_type};base64,{encoded}")

    model = load_model(agent_name)
    if _provider() == "ollama":
        from langchain_core.messages import HumanMessage

        return model.invoke([HumanMessage(content=[
            {"type": "text", "text": prompt},
            *({"type": "image_url", "image_url": image} for image in images),
        ])])
    return model.invoke([
        {"type": "text", "text": prompt},
        *({"type": "image_url", "image_url": {"url": image}} for image in images),
    ])


def invoke_with_images_and_evaluation(agent_name: str, prompt: str, image_files: list[str], *, purpose: str):
    started = time.monotonic()
    response = invoke_with_images(agent_name, prompt, image_files)
    return response, _evaluation(
        response, prompt, agent_name, purpose, model_name_for(agent_name),
        elapsed_seconds=time.monotonic() - started,
    )
