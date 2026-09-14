# Magnific Video Automation Agent

A LangGraph pipeline that turns a story prompt into a narrated short video. It plans the story and shots, generates reference art and scene media, validates each stage, assembles a rough cut, and produces the final edit.

## Demo

[Watch the robot story demo](demo/robot-story.mp4) — 17 seconds, 1080×1920, H.264/AAC.

## Pipeline

```text
Prompt → safety check → story → narration → scenes → visual beats → shots
→ reference art → storyboard → image QA → motion → shot videos
→ video validation → edit timeline → rough cut → combined judge
→ subtitles → final edit
```

Human review checkpoints and SQLite persistence allow interrupted runs to be reviewed or resumed. Detailed agent contracts are in [`docs/agents`](docs/agents/README.md).

## Requirements

- Python 3.10+
- FFmpeg and ffprobe
- Ollama with text and vision models
- Magnific API key for image/video generation
- ElevenLabs API key for narration
- MinIO only when externally reachable source media is required

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API keys to `.env`, then install the default local models:

```bash
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5vl:3b
ollama serve
```

## Run the web app

In a second terminal:

```bash
source .venv/bin/activate
python backend.py
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001), enter a story prompt, and start generation. Artifacts are written under `outputs/<thread-id>/`.

## Run the interactive CLI

```bash
source .venv/bin/activate
python main.py
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite currently contains 189 tests.
