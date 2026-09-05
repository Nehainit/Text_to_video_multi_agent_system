import story_agent


class FakeModel:
    def invoke(self, prompt):
        assert "Raja" in prompt
        return type(
            "Response",
            (),
            {
                "content": (
                    '{"story": "Raja guarded a mango basket while Chuha watched from the doorway. '
                    "When the basket tipped over, Chuha caught the last mango before it rolled into the mud. "
                    "Raja split it with him, and the picnic became funnier and warmer than eating alone.\"}"
                )
            },
        )()


def test_create_story():
    original = story_agent.load_model
    story_agent.load_model = lambda agent_name: FakeModel()
    try:
        result = story_agent.create_story(
            {
                "topic": "sharing",
                "tone": "funny",
                "duration": "30 seconds",
                "language": "English",
                "characters": ["Raja"],
            }
        )
    finally:
        story_agent.load_model = original

    assert "picnic became funnier" in result["story"]


def test_rejects_planning_data():
    class BadModel:
        def invoke(self, prompt):
            return type("Response", (), {"content": '{"frames": []}'})()

    original = story_agent.load_model
    story_agent.load_model = lambda agent_name: BadModel()
    try:
        try:
            story_agent.create_story(
                {
                    "topic": "sharing",
                    "tone": "funny",
                    "duration": "30 seconds",
                    "language": "English",
                    "characters": ["Raja"],
                }
            )
        except RuntimeError:
            return
    finally:
        story_agent.load_model = original

    raise AssertionError("planning data should fail")


def test_retries_after_bad_story():
    class RetryModel:
        calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return type("Response", (), {"content": '{"story": "Raja, a monkey, had food. Chuha came. Good."}'})()
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"story": "Raja saved the last roti for himself while Chuha watched quietly. '
                        "When the plate slipped, Chuha caught it before the food fell. "
                        "Raja shared the roti, and their small meal ended with both laughing together.\"}"
                    )
                },
            )()

    model = RetryModel()
    original = story_agent.load_model
    story_agent.load_model = lambda agent_name: model
    try:
        result = story_agent.create_story(
            {
                "topic": "sharing",
                "tone": "funny",
                "duration": "30 seconds",
                "language": "English",
                "characters": ["Raja", "Chuha"],
            }
        )
    finally:
        story_agent.load_model = original

    assert model.calls == 2
    assert "both laughing together" in result["story"]


def test_rejects_abrupt_story():
    try:
        story_agent._validate_story(
            "Raja had food. Chuha was hungry. Done.",
            {
                "topic": "sharing",
                "tone": "funny",
                "duration": "30 seconds",
                "language": "English",
                "characters": ["Raja"],
            },
        )
    except RuntimeError:
        return
    raise AssertionError("abrupt story should fail")


def test_rejects_invented_identity():
    try:
        story_agent._validate_story(
            "Raja, a tiny monkey, found a mango. Chuha asked for a bite. Raja shared it, and both felt proud.",
            {
                "topic": "sharing",
                "tone": "funny",
                "duration": "30 seconds",
                "language": "English",
                "characters": ["Raja", "Chuha"],
            },
        )
    except RuntimeError:
        return
    raise AssertionError("invented character identity should fail")


def test_accepts_hindi_sentence_breaks():
    story_agent._validate_story(
        "राजा और रानी शहर में चले। लोगों की परेशानी देखकर वे रुक गए। दोनों ने मदद की और सीखा कि सच्चा राजधर्म सेवा है।",
        {
            "topic": "city care",
            "tone": "concerned",
            "duration": "30 seconds",
            "language": "Hindi",
            "characters": "Raja, Rani",
        },
    )


if __name__ == "__main__":
    test_create_story()
    test_rejects_planning_data()
    test_retries_after_bad_story()
    test_rejects_abrupt_story()
    test_rejects_invented_identity()
    test_accepts_hindi_sentence_breaks()
