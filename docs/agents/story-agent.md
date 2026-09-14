# Story Agent

## Goal

Take the user's request and return a reviewed story idea and structure together
with parsed requirements. This stage stops at JSON; it does not produce narration
audio, images, video, or a compiled film.

## Input

`create_story(user_input)` accepts either:

- A nonempty string containing the user's request.
- A dictionary with a nonempty `topic` and optional `duration`, `language`,
  `tone`, `characters`, `visual_style`, `must_include`, `must_avoid`, and
  `other_constraints`.

Existing workflow calls may also supply a previous draft and `review_note`.
Supplying previously parsed `parsed_requirements` skips requirement extraction
and starts directly with story-only generation.

## Output

The returned dictionary is JSON-serializable and includes per-call evaluation:

```json
{
  "story": {
    "idea": "...",
    "characters": [
      {"character_id": "character-001", "name": "...", "description": "..."}
    ],
    "structure": [
      {"beat_id": "beat-001", "description": "..."},
      {"beat_id": "beat-002", "description": "..."},
      {"beat_id": "beat-003", "description": "..."}
    ]
  },
  "parsed_requirements": {
    "topic": "...",
    "duration_seconds": null,
    "language": null,
    "characters": [],
    "tone": null,
    "visual_style": null,
    "must_include": [],
    "must_avoid": [],
    "other_constraints": []
  },
  "llm_evaluations": [
    {
      "call_number": 1,
      "agent": "story",
      "purpose": "generate_story_and_requirements",
      "model": "configured-model",
      "input_tokens": 500,
      "output_tokens": 150,
      "total_tokens": 650,
      "token_source": "provider",
      "cost_usd": 0.00025
    }
  ]
}
```

Requirements come from the original request, not invented story details.
Unspecified scalar requirements remain `null`; unspecified lists remain empty.
An explicit duration is converted to a positive integer number of seconds.
Generated characters contain only `name` and `description`, and generated beats contain only
`description`. The backend adds consecutive character and beat IDs beginning at
`character-001` and `beat-001`. Every story has at least setup, meaningful change, and resolution;
longer stories may add more beats.

## Story Review Agent

Each structurally valid candidate is passed to
[`story_review_agent.py`](../../story_review_agent.py):

```python
review_story(parsed_requirements, story)
```

The reviewer accepts the saved requirements and the current story's `idea`, `characters`, and
`structure`. It verifies that every recurring visible story character is listed. A human revision note can also be supplied as a keyword argument.
It uses the existing story model in a separate call and
returns approval, issues, and that call's `llm_evaluation`.

The reviewer compares the idea and beats against the original input and parsed
requirements, including constraints and exclusions.
It checks a complete story arc and reasonable scope for the requested duration.
Exact narration word counts and speaking time belong to later stages.

## Content safety

Before the first story-generation call, a safety-model call classifies the user
request. Requests for sexual content, self-harm, graphic violence, hate or
harassment, abuse or exploitation, or dangerous and illegal instructions are
rejected before generation. Invalid classifier output also stops the request.
The API returns HTTP 400 with the matched categories for a blocked request.

The policy, categories, classifier prompt, and enforcement actions live in
[`config/content_safety.yml`](../../config/content_safety.yml). The Story Agent
prompt carries the same family-safe policy. The existing Story
Review Agent checks the generated idea and every beat against that policy; an
unsafe candidate is rejected with actionable feedback and regenerated. The
input-safety call is included in `llm_evaluations`. `SAFETY_MODEL` can select a
dedicated classifier and otherwise uses the configured default model.

The first generation returns `story` plus `parsed_requirements`. Once valid
requirements have been saved, every subsequent generation returns only:

```json
{"story": {"idea": "...", "characters": [{"name": "...", "description": "..."}], "structure": [{"description": "..."}]}}
```

A rejected candidate is regenerated using the previous draft, cached parsed
requirements, and the reviewer's specific feedback. Raw user input is sent only
on the first call. The generator must not return or re-extract requirements on
retries. Code attaches the saved requirements to the updated story before each
review, so the reviewer always receives both. Successful `create_story` calls
return all generation and review evaluations in execution order.

Provider token metadata is used when available. When a provider omits it, token
counts are approximate and `token_source` is `estimated`. Configure
`LLM_INPUT_USD_PER_1M_TOKENS` and `LLM_OUTPUT_USD_PER_1M_TOKENS` to calculate
`cost_usd`; without both prices, `cost_usd` is `null` rather than a false zero.

The agent returns only after an explicit approval with no outstanding issues.
Invalid JSON, malformed fields, or failed reviews never count as approval.
After three unsuccessful generation attempts, it raises an error containing
the latest feedback.

The semantic review is model-based; it is not a guarantee that every requirement
will be interpreted correctly.

## Existing workflow integration

`create_story_node` adapts this result for the existing video graph: it joins
beat descriptions into the prose `story` consumed downstream and retains the
structured story in `story_outline`, alongside `parsed_requirements`.
The existing human story review still follows that node. Direct calls to
`create_story` include the evaluation list shown above.

The standalone entry point in `story_agent.py` accepts a prompt argument,
interactive input, or text from standard input and prints JSON only on success.
It stops after the story agent.

## Validation

Run the local checks without live model calls:

```sh
.venv/bin/python -m pytest test_story_agent.py test_story_hitl_agent.py test_preproduction_agents.py test_backend.py -q
```

## Implementation

[`story_agent.py`](../../story_agent.py)
