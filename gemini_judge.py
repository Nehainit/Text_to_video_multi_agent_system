import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from types import SimpleNamespace

from models import _evaluation, _request_timeout_seconds
from prompts import JUDGE_PANEL_CONFIG
from story_agent import _parse_json


logger = logging.getLogger(__name__)


def _google_sdk():
    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install google-genai to run Gemini video judging.") from exc
    return genai, types


def _wait_for_file(client, uploaded, processing_config=None):
    config = processing_config or JUDGE_PANEL_CONFIG
    deadline = time.monotonic() + float(config["file_processing_timeout_seconds"])
    while str(getattr(getattr(uploaded, "state", None), "name", "")).upper() == "PROCESSING":
        if time.monotonic() >= deadline:
            raise RuntimeError("Gemini timed out while processing the uploaded video.")
        time.sleep(float(config["file_processing_poll_seconds"]))
        uploaded = client.files.get(name=uploaded.name)
    if str(getattr(getattr(uploaded, "state", None), "name", "")).upper() == "FAILED":
        raise RuntimeError("Gemini failed to process the uploaded video.")
    return uploaded


def _usage_response(response):
    usage = getattr(response, "usage_metadata", None)
    metadata = {
        "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
    } if usage else None
    return SimpleNamespace(content=getattr(response, "text", ""), usage_metadata=metadata)


def invoke_gemini_judge(
    *,
    agent_name: str,
    purpose: str,
    system_prompt: str,
    response_schema: dict,
    payload: dict,
    video_file: str | None = None,
    image_files: list[str] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    processing_config: dict | None = None,
) -> tuple[object, dict]:
    images = image_files or []
    if video_file and not Path(video_file).is_file():
        raise RuntimeError(f"{agent_name} video is missing: {video_file}")
    for image_file in images:
        if not Path(image_file).is_file():
            raise RuntimeError(f"{agent_name} reference image is missing: {image_file}")

    genai, types = _google_sdk()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to run Gemini video judging.")
    model = model or os.getenv("QA_MODEL") or os.getenv("GEMINI_JUDGE_MODEL") or JUDGE_PANEL_CONFIG["model"]
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(_request_timeout_seconds() * 1000)),
    )
    uploaded = None
    try:
        contents = []
        if video_file:
            uploaded = _wait_for_file(client, client.files.upload(file=video_file), processing_config)
            contents.append(uploaded)
        contents.append(json.dumps(payload, ensure_ascii=False))
        contents.extend(types.Part.from_bytes(
            data=Path(image_file).read_bytes(),
            mime_type=mimetypes.guess_type(image_file)[0] or "image/jpeg",
        ) for image_file in images)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=float(JUDGE_PANEL_CONFIG["temperature"] if temperature is None else temperature),
                response_mime_type="application/json",
                response_json_schema=response_schema,
            ),
        )
        parsed = getattr(response, "parsed", None) or _parse_json(response.text)
        evaluation = _evaluation(_usage_response(response), payload, agent_name, purpose, model)
        return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed, evaluation
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                logger.warning("%s could not delete Gemini file %s", agent_name, uploaded.name)
        client.close()
