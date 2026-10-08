import httpx
import pytest

from docplainer.config import Settings
from docplainer.generation.llm import FallbackChain, LLMError, OpenAICompatClient, build_chain


def client(name="groq") -> OpenAICompatClient:
    return OpenAICompatClient(name, "https://x.test/v1", "key", "m", max_retries=1)


def resp(status, json=None, headers=None):
    return httpx.Response(
        status, json=json, headers=headers, request=httpx.Request("POST", "https://x.test")
    )


OK = {"choices": [{"message": {"content": "hello"}}]}


def test_complete_returns_text(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda **kw: resp(200, OK))
    assert client().complete([{"role": "user", "content": "hi"}]) == "hello"


def test_retries_rate_limit_then_succeeds(monkeypatch):
    seen = iter([resp(429, headers={"retry-after": "0"}), resp(200, OK)])
    monkeypatch.setattr(httpx, "post", lambda **kw: next(seen))
    assert client().complete([]) == "hello"


def test_persistent_rate_limit_raises_retryable(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda **kw: resp(429, headers={"retry-after": "0"}))
    with pytest.raises(LLMError) as e:
        client().complete([])
    assert e.value.retryable


def test_fallback_moves_to_next_provider_and_records_which(monkeypatch):
    def post(**kw):
        return (
            resp(429, headers={"retry-after": "0"})
            if "groq" in kw["json"]["model"]
            else resp(200, OK)
        )

    a = OpenAICompatClient("groq", "https://x.test/v1", "k", "groq-model", max_retries=0)
    b = OpenAICompatClient("gemini", "https://y.test/v1", "k", "gemini-model", max_retries=0)
    monkeypatch.setattr(httpx, "post", post)
    chain = FallbackChain([a, b])
    assert chain.complete([]) == "hello" and chain.last_used == "gemini"


def test_all_providers_failing_gives_one_clear_error(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda **kw: resp(500))
    chain = FallbackChain([client("groq"), client("gemini")])
    with pytest.raises(LLMError, match="All providers failed"):
        chain.complete([])


def test_stream_fallback_only_before_first_token():
    class Dead:
        name = "dead"

        def stream(self, *a, **k):
            raise LLMError("down")
            yield  # pragma: no cover

    class Live:
        name = "live"

        def stream(self, *a, **k):
            yield "a"
            yield "b"

    chain = FallbackChain([Dead(), Live()])  # type: ignore[list-item]
    assert "".join(chain.stream([])) == "ab" and chain.last_used == "live"


def test_build_chain_skips_providers_without_keys():
    chain = build_chain({"groq": "k", "gemini": None, "openrouter": ""}, Settings())
    assert [c.name for c in chain.clients] == ["groq"]
    with pytest.raises(ValueError):
        build_chain({}, Settings())
