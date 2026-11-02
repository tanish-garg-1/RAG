import pytest

from src.generation.llm_router import LLMRouter


class FakeClient:
    def __init__(self, reply="ok", error: Exception | None = None):
        self.reply, self.error = reply, error
        self.calls = 0
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 2}

    def complete(self, system_prompt, user_prompt, temperature=0.2):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply


def router(primary, backup):
    return LLMRouter([("groq", lambda: primary), ("azure", lambda: backup)])


def test_primary_answers_and_backup_is_untouched():
    primary, backup = FakeClient("from groq"), FakeClient("from azure")
    r = router(primary, backup)
    assert r.complete("sys", "hi") == "from groq"
    assert backup.calls == 0
    assert r.last_provider == "groq" and r.stats["fallbacks"] == 0


def test_primary_failure_falls_back_to_backup():
    r = router(FakeClient(error=TimeoutError("slow")), FakeClient("from azure"))
    assert r.complete("sys", "hi") == "from azure"
    assert r.last_provider == "azure"
    assert r.stats["fallbacks"] == 1
    assert r.stats["by_provider"]["groq"]["error"] == 1
    assert r.last_usage == {"prompt_tokens": 1, "completion_tokens": 2}


def test_unconfigured_backup_gives_a_clear_error_when_primary_fails():
    def not_configured():
        raise RuntimeError("Azure OpenAI is not configured")

    r = LLMRouter([("groq", lambda: FakeClient(error=RuntimeError("429"))), ("azure", not_configured)])
    with pytest.raises(RuntimeError, match="All LLM providers failed.*groq.*azure"):
        r.complete("sys", "hi")
    assert r.stats["failures"] == 1


def test_unconfigured_primary_skips_to_backup():
    def no_key():
        raise RuntimeError("GROQ_API_KEY is not set")

    r = LLMRouter([("groq", no_key), ("azure", lambda: FakeClient("from azure"))])
    assert r.complete("sys", "hi") == "from azure"


def test_temperature_is_forwarded_only_when_given():
    seen = {}

    class Spy(FakeClient):
        def complete(self, system_prompt, user_prompt, **kwargs):
            seen.update(kwargs)
            return "x"

    r = LLMRouter([("groq", lambda: Spy())])
    r.complete("s", "u")
    assert seen == {}
    r.complete("s", "u", temperature=0)
    assert seen == {"temperature": 0}
