import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from video_automation.agents import story_agent
from video_automation.agents import story_review_agent


USER_INPUT = "Create a 30-second story where Raja and Chuha share a mango. Include a golden bell. No violence."
SAFE = {"safe": True, "categories": [], "reason": None}


def prompt_text(prompt):
    return prompt if isinstance(prompt, str) else "\n".join(message["content"] for message in prompt)


def story_result():
    return {
        "story": {
            "idea": "Sharing a mango leads Raja and Chuha to a musical surprise.",
            "characters": [
                {"character_id": "character-001", "name": "Raja", "description": "Raja"},
                {"character_id": "character-002", "name": "Chuha", "description": "Chuha"},
            ],
            "structure": [
                {"beat_id": "beat-001", "description": "Raja finds a mango that Chuha cannot reach."},
                {"beat_id": "beat-002", "description": "Raja and Chuha work together and decide to share the mango."},
                {"beat_id": "beat-003", "description": "They discover a golden bell inside and celebrate together."},
            ],
        },
        "parsed_requirements": {
            "topic": "Raja and Chuha share a mango",
            "duration_seconds": 30,
            "language": None,
            "characters": ["Raja", "Chuha"],
            "tone": None,
            "visual_style": None,
            "must_include": ["a golden bell"],
            "must_avoid": ["violence"],
            "other_constraints": [],
        },
    }


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        return SimpleNamespace(content=response if isinstance(response, str) else json.dumps(response, ensure_ascii=False))


def assert_story_result(result, expected, calls):
    assert result["story"] == expected["story"]
    assert result["parsed_requirements"] == expected["parsed_requirements"]
    assert len(result["llm_evaluations"]) == calls
    assert [item["call_number"] for item in result["llm_evaluations"]] == list(range(1, calls + 1))
    assert all(item["input_tokens"] > 0 and item["output_tokens"] > 0 for item in result["llm_evaluations"])


def test_story_returns_exact_contract_after_internal_review():
    expected = story_result()
    model = FakeModel([SAFE, expected, {"approved": True, "issues": []}])
    with patch.object(story_agent, "load_model", return_value=model), patch.object(story_review_agent, "load_model", return_value=model):
        result = story_agent.create_story(USER_INPUT)

    assert_story_result(result, expected, 3)
    assert set(result) == {"story", "parsed_requirements", "llm_evaluations"}
    assert [item["purpose"] for item in result["llm_evaluations"]] == [
        "review_user_input_safety", "generate_story_and_requirements", "review_story_against_requirements"
    ]
    assert len(model.prompts) == 3
    assert USER_INPUT in prompt_text(model.prompts[0])
    assert USER_INPUT in prompt_text(model.prompts[1])
    assert USER_INPUT not in prompt_text(model.prompts[2])
    assert "Story Review Agent" in prompt_text(model.prompts[2])
    assert result["parsed_requirements"]["duration_seconds"] == 30


def test_review_feedback_regenerates_story_with_fixed_requirements():
    first = story_result()
    first["story"]["structure"][2]["description"] = "They share the mango and go home."
    revised = story_result()
    issue = "Show the requested golden bell in beat-003."
    generator = FakeModel([SAFE, first, {"story": revised["story"]}])
    reviewer = FakeModel([{"approved": False, "issues": [issue]}, {"approved": True, "issues": []}])
    with patch.object(story_agent, "load_model", return_value=generator), patch.object(story_review_agent, "load_model", return_value=reviewer):
        result = story_agent.create_story(USER_INPUT)

    assert_story_result(result, revised, 5)
    assert [item["purpose"] for item in result["llm_evaluations"]] == [
        "review_user_input_safety", "generate_story_and_requirements", "review_story_against_requirements",
        "regenerate_story", "review_story_against_requirements",
    ]
    assert len(generator.prompts) == 3 and len(reviewer.prompts) == 2
    assert issue in prompt_text(generator.prompts[2])
    assert json.dumps(first["story"]) in prompt_text(generator.prompts[2])
    assert json.dumps(first["parsed_requirements"]) in prompt_text(generator.prompts[2])
    assert '"parsed_requirements":' in prompt_text(generator.prompts[1])
    assert '"parsed_requirements":' not in prompt_text(generator.prompts[2])
    assert "Return only story" in prompt_text(generator.prompts[2])
    assert json.dumps(first) in reviewer.prompts[0]
    assert json.dumps(revised) in reviewer.prompts[1]
    assert USER_INPUT not in prompt_text(generator.prompts[2])
    assert "sexual content" in reviewer.prompts[0] and "self-harm" in reviewer.prompts[0]
    assert all(USER_INPUT not in prompt for prompt in reviewer.prompts)


def test_regeneration_cannot_change_requirements():
    original = story_result()
    changed = story_result()
    changed["parsed_requirements"]["must_include"] = []
    model = FakeModel([
        SAFE, original, {"approved": False, "issues": ["Improve the ending."]},
        changed,
        {"story": original["story"]}, {"approved": True, "issues": []},
    ])
    with patch.object(story_agent, "load_model", return_value=model), patch.object(story_review_agent, "load_model", return_value=model):
        result = story_agent.create_story(USER_INPUT)

    assert result["parsed_requirements"] == original["parsed_requirements"]
    assert len(model.prompts) == 6
    assert "Return only story" in prompt_text(model.prompts[4])


def test_malformed_output_is_retried_and_never_returned():
    wrong_beat_key = story_result()
    wrong_beat_key["story"]["structure"][0] = {"beat": 1, "description": "Opening."}
    missing_requirements = story_result()
    del missing_requirements["parsed_requirements"]["must_avoid"]
    invalid_duration = story_result()
    invalid_duration["parsed_requirements"]["duration_seconds"] = True
    invalid_list = story_result()
    invalid_list["parsed_requirements"]["characters"] = "Raja"
    for invalid in (
        "not JSON", [], {"frames": []}, wrong_beat_key,
        missing_requirements, invalid_duration, invalid_list,
    ):
        model = FakeModel([SAFE, invalid, invalid, invalid])
        with patch.object(story_agent, "load_model", return_value=model), patch.object(story_review_agent, "load_model", return_value=model):
            try:
                story_agent.create_story(USER_INPUT)
            except RuntimeError as exc:
                assert "after 3 attempts" in str(exc)
            else:
                raise AssertionError("Invalid story output was accepted")
        assert len(model.prompts) == 4
        assert "Internal review feedback:" in prompt_text(model.prompts[-1])


def test_story_beat_ids_are_assigned_by_backend():
    generated = story_result()
    for beat in generated["story"]["structure"]:
        beat["beat_id"] = "wrong"
    model = FakeModel([SAFE, generated, {"approved": True, "issues": []}])
    with patch.object(story_agent, "load_model", return_value=model), patch.object(story_review_agent, "load_model", return_value=model):
        result = story_agent.create_story(USER_INPUT)

    assert result["story"] == story_result()["story"]


def test_story_character_ids_are_assigned_by_backend():
    generated = story_result()
    for character in generated["story"]["characters"]:
        character["character_id"] = "wrong"
    model = FakeModel([SAFE, generated, {"approved": True, "issues": []}])
    with patch.object(story_agent, "load_model", return_value=model), patch.object(story_review_agent, "load_model", return_value=model):
        result = story_agent.create_story(USER_INPUT)

    assert [character["character_id"] for character in result["story"]["characters"]] == [
        "character-001", "character-002"
    ]


def test_internal_review_fails_closed_and_stops_after_three_attempts():
    for review in (
        {"approved": False, "issues": ["The required golden bell is missing."]},
        {"approved": True, "issues": ["The required golden bell is missing."]},
        {"approved": "true", "issues": []},
        {"approved": False, "issues": []},
        {"approved": True},
        "not JSON",
    ):
        model = FakeModel([SAFE, story_result(), review, {"story": story_result()["story"]}, review, {"story": story_result()["story"]}, review])
        with patch.object(story_agent, "load_model", return_value=model), patch.object(story_review_agent, "load_model", return_value=model):
            try:
                story_agent.create_story(USER_INPUT)
            except RuntimeError as exc:
                assert "after 3 attempts" in str(exc)
                if isinstance(review, dict) and review.get("issues"):
                    assert "golden bell" in str(exc)
            else:
                raise AssertionError("An unapproved story was returned")
        assert len(model.prompts) == 7


def test_user_input_validation_and_structured_request():
    with patch.object(story_agent, "load_model") as loader:
        for invalid in ("", "  ", None, [], {}, {"topic": 7}):
            try:
                story_agent.create_story(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Empty or invalid user input was accepted")
        loader.assert_not_called()

    user_input = {
        "topic": USER_INPUT, "duration": "30 seconds", "language": "Hindi",
        "characters": ["Raja", "Chuha"], "thread_id": "internal-session",
    }
    original_input = deepcopy(user_input)
    expected = story_result()
    expected["parsed_requirements"].update(duration_seconds=30, language="Hindi")
    expected["story"] = {
        "idea": "राजा और चूहा आम बाँटकर एक सुनहरी घंटी खोजते हैं।",
        "characters": [
            {"character_id": "character-001", "name": "Raja", "description": "Raja"},
            {"character_id": "character-002", "name": "Chuha", "description": "Chuha"},
        ],
        "structure": [
            {"beat_id": "beat-001", "description": "राजा को एक आम मिलता है, जिसे चूहा नहीं पहुँच पाता।"},
            {"beat_id": "beat-002", "description": "राजा और चूहा मिलकर आम बाँटने का फैसला करते हैं।"},
            {"beat_id": "beat-003", "description": "दोनों आम में सुनहरी घंटी पाकर साथ खुश होते हैं।"},
        ],
    }
    model = FakeModel([SAFE, expected, {"approved": True, "issues": []}])
    with patch.object(story_agent, "load_model", return_value=model), patch.object(story_review_agent, "load_model", return_value=model):
        result = story_agent.create_story(user_input)

    assert_story_result(result, expected, 3)
    assert user_input == original_input
    assert "30 seconds" in prompt_text(model.prompts[1]) and "Hindi" in prompt_text(model.prompts[1])
    assert '"duration_seconds": 30' in prompt_text(model.prompts[2]) and "Hindi" in prompt_text(model.prompts[2])
    assert all("internal-session" not in prompt_text(prompt) for prompt in model.prompts)
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result


def test_unsafe_input_is_blocked_before_story_generation():
    model = FakeModel([{
        "safe": False,
        "categories": ["sexual_content", "self_harm"],
        "reason": "The requested story contains prohibited content.",
    }])
    with patch.object(story_agent, "load_model", return_value=model), patch.object(story_review_agent, "load_model") as reviewer:
        try:
            story_agent.create_story("Create an explicit sexual story involving self-harm.")
        except story_agent.UnsafeContentError as exc:
            assert exc.categories == ["sexual_content", "self_harm"]
        else:
            raise AssertionError("Unsafe input reached story generation")

    assert len(model.prompts) == 1
    assert model.prompts[0][0]["role"] == "system"
    reviewer.assert_not_called()


def test_invalid_safety_decision_fails_closed():
    model = FakeModel([{"safe": True, "categories": ["self_harm"], "reason": None}])
    with patch.object(story_agent, "load_model", return_value=model):
        try:
            story_agent.create_story(USER_INPUT)
        except RuntimeError as exc:
            assert "invalid decision" in str(exc)
        else:
            raise AssertionError("Invalid safety output reached story generation")
    assert len(model.prompts) == 1


def test_safety_accepts_semantically_safe_json_variants():
    model = FakeModel([{"safe": "true", "categories": ["family", "adventure"], "reason": "No prohibited content detected."}])
    with patch.object(story_agent, "load_model", return_value=model):
        assert story_agent._check_input_safety({"topic": "A friendly picnic story"})


def test_unsafe_generated_story_is_rejected_and_regenerated():
    unsafe = story_result()
    unsafe["story"]["structure"][1]["description"] = "A character attempts self-harm."
    safe = story_result()
    issue = "self_harm in beat-002 violates the family-safe policy."
    generator = FakeModel([SAFE, unsafe, {"story": safe["story"]}])
    reviewer = FakeModel([
        {"approved": False, "issues": [issue]},
        {"approved": True, "issues": []},
    ])
    with patch.object(story_agent, "load_model", return_value=generator), patch.object(story_review_agent, "load_model", return_value=reviewer):
        result = story_agent.create_story(USER_INPUT)

    assert result["story"] == safe["story"]
    assert issue in prompt_text(generator.prompts[2])
    assert len(result["llm_evaluations"]) == 5


def test_human_revision_and_existing_graph_handoff():
    from video_automation import agent_graph
    from langgraph.types import Command

    initial = story_result()
    revised = story_result()
    revised["story"]["structure"][1]["description"] += " The bell answers with unexpected music."
    model = FakeModel([
        SAFE, initial, {"approved": True, "issues": []}, SAFE,
        {"story": initial["story"]},
        {"story": revised["story"]}, {"approved": True, "issues": []},
    ])
    state = {"topic": USER_INPUT, "duration": "30 seconds", "language": "English", "tone": "", "characters": []}
    graph = agent_graph.build_story_graph()
    config = {"configurable": {"thread_id": "story-contract-test"}}
    with patch.object(story_agent, "load_model", return_value=model), patch.object(story_review_agent, "load_model", return_value=model):
        result = graph.invoke(state, config)
        assert result["story_outline"] == initial["story"]
        assert result["parsed_requirements"] == initial["parsed_requirements"]
        assert result["story_beats"] == initial["story"]["structure"]
        assert isinstance(result["__interrupt__"][0].value["story"], str)
        result = graph.invoke(Command(resume={"action": "regenerate", "note": "Make the ending surprising."}), config)

    assert result["story_outline"] == revised["story"]
    assert "unexpected music" in result["story"]
    assert result["review_note"] == ""
    assert "Make the ending surprising." in prompt_text(model.prompts[4])
    assert "unchanged story" in prompt_text(model.prompts[5])
    assert '"parsed_requirements":' not in prompt_text(model.prompts[4])


def test_story_review_agent_accepts_requirements_and_story_directly():
    candidate = story_result()
    original = deepcopy(candidate)
    for verdict in (
        {"approved": True, "issues": []},
        {"approved": False, "issues": ["The requested event is missing from beat-002."]},
    ):
        reviewer = FakeModel([verdict])
        with patch.object(story_review_agent, "load_model", return_value=reviewer):
            result = story_review_agent.review_story(candidate["parsed_requirements"], candidate["story"])
        assert result["approved"] == verdict["approved"]
        assert result["issues"] == verdict["issues"]
        assert result["llm_evaluation"]["purpose"] == "review_story_against_requirements"
        assert candidate == original
        assert len(reviewer.prompts) == 1
        assert json.dumps(candidate) in reviewer.prompts[0]

    with patch.object(story_review_agent, "load_model") as loader:
        for requirements, story in (({}, candidate["story"]), (candidate["parsed_requirements"], {})):
            try:
                story_review_agent.review_story(requirements, story)
            except ValueError:
                pass
            else:
                raise AssertionError("The reviewer accepted invalid input")
        loader.assert_not_called()


def test_ollama_story_review_ignores_vague_rejection_but_keeps_material_failures():
    ChatOllama = type("ChatOllama", (FakeModel,), {})
    candidate = story_result()
    for issue, approved in (
        ("The robot wanders while trying to remember its home.", True),
        ("The required ending is missing.", False),
    ):
        reviewer = ChatOllama([{"approved": False, "issues": [issue]}])
        with patch.object(story_review_agent, "load_model", return_value=reviewer):
            result = story_review_agent.review_story(candidate["parsed_requirements"], candidate["story"])
        assert result["approved"] is approved


def test_saved_requirements_are_reused_without_extracting_again():
    expected = story_result()
    user_input = {"topic": USER_INPUT, "parsed_requirements": expected["parsed_requirements"]}
    original_input = deepcopy(user_input)
    generator = FakeModel([{"story": expected["story"]}])
    reviewer = FakeModel([{"approved": True, "issues": []}])
    with patch.object(story_agent, "load_model", return_value=generator), patch.object(story_review_agent, "load_model", return_value=reviewer):
        result = story_agent.create_story(user_input)
    assert_story_result(result, expected, 2)
    assert user_input == original_input
    assert '"parsed_requirements":' not in generator.prompts[0]
    assert json.dumps(expected) in reviewer.prompts[0]


if __name__ == "__main__":
    test_story_returns_exact_contract_after_internal_review()
    test_review_feedback_regenerates_story_with_fixed_requirements()
    test_regeneration_cannot_change_requirements()
    test_malformed_output_is_retried_and_never_returned()
    test_internal_review_fails_closed_and_stops_after_three_attempts()
    test_user_input_validation_and_structured_request()
    test_human_revision_and_existing_graph_handoff()
    test_story_review_agent_accepts_requirements_and_story_directly()
    test_saved_requirements_are_reused_without_extracting_again()
