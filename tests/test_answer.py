from docplainer.config import Settings
from docplainer.generation import AnswerService
from docplainer.ingestion import ingest_files
from docplainer.retrieval import HybridIndex, Retriever
from docplainer.retrieval.embeddings import HashingEmbedder
from docplainer.schemas import Chunk, Hit


class FakeLLM:
    def __init__(self, reply="Europe made 141.9 million EUR [1]."):
        self.reply, self.calls = reply, []

    def complete(self, messages, **kw):
        self.calls.append(messages)
        return self.reply

    def stream(self, messages, **kw):
        yield from self.reply.split(" ")


def web_hit(text="Canberra is the capital of Australia."):
    c = Chunk("w", "web", "Wikipedia", 0, "", "web", text, text, url="https://w.test")
    return Hit(c, 1.0, 1.0)


def make(sample_dir, llm, web=lambda q: [], confident=True):
    cfg = Settings(
        embedder_backend="hashing", use_reranker=False, min_dense_cosine=0.0 if confident else 0.99
    )
    idx = HybridIndex(HashingEmbedder())
    idx.add(ingest_files([sample_dir / "northwind_annual_report.pdf"], cfg).chunks)
    return AnswerService(Retriever(idx, None, cfg), llm, cfg, search_web=web)


def test_grounded_answer_returns_only_cited_passages(sample_dir):
    llm = FakeLLM("Europe revenue was 141.9 [1].")
    svc = make(sample_dir, llm)
    ans = svc.answer("What was Europe revenue in FY2024?")
    assert ans.mode == "documents"
    assert [c.n for c in ans.citations] == [1]
    assert ans.citations[0].label.startswith("northwind_annual_report.pdf p.")
    system, user = llm.calls[0]
    assert (
        "Context passages" in user["content"]
        and "[1] northwind_annual_report.pdf" in user["content"]
    )
    assert "untrusted" in system["content"]


def test_invented_citation_numbers_are_ignored(sample_dir):
    ans = make(sample_dir, FakeLLM("Made up [99].")).answer("Europe revenue?")
    assert ans.citations and all(c.n != 99 for c in ans.citations)


def test_low_confidence_falls_back_to_web_and_labels_it(sample_dir):
    llm = FakeLLM("The documents don't cover it; from the web: Canberra [1].")
    svc = make(sample_dir, llm, web=lambda q: [web_hit()], confident=False)
    ans = svc.answer("What is the capital of Australia?")
    assert ans.mode == "web" and ans.citations[0].url == "https://w.test"
    assert "web search" in llm.calls[0][1]["content"]


def test_low_confidence_without_web_refuses_without_calling_the_llm(sample_dir):
    llm = FakeLLM()
    svc = make(sample_dir, llm, confident=False)
    ans = svc.answer("What is the capital of Australia?", allow_web=False)
    assert ans.mode == "none" and "couldn't find" in ans.text and llm.calls == []


def test_web_fallback_empty_results_also_refuses(sample_dir):
    ans = make(sample_dir, FakeLLM(), web=lambda q: [], confident=False).answer("capital?")
    assert ans.mode == "none"


def test_no_documents_prompts_upload():
    cfg = Settings(embedder_backend="hashing")
    svc = AnswerService(Retriever(HybridIndex(HashingEmbedder()), None, cfg), FakeLLM(), cfg)
    ans = svc.answer("anything", allow_web=False)
    assert ans.mode == "none" and "Upload" in ans.text


def test_small_talk_costs_no_llm_call(sample_dir):
    llm = FakeLLM()
    ans = make(sample_dir, llm).answer("thanks!")
    assert ans.mode == "chat" and llm.calls == []


def test_followup_is_rewritten_but_first_question_is_not(sample_dir):
    llm = FakeLLM("revenue in Europe in FY2023")
    svc = make(sample_dir, llm)
    assert svc.standalone_query("Europe revenue?", []) == "Europe revenue?" and llm.calls == []
    hist = [
        {"role": "user", "content": "Europe revenue FY2024?"},
        {"role": "assistant", "content": "141.9"},
    ]
    assert svc.standalone_query("and 2023?", hist) == "revenue in Europe in FY2023"
    assert "History:" in llm.calls[0][1]["content"]


def test_rewrite_failure_degrades_to_original_question(sample_dir):
    from docplainer.generation.llm import LLMError

    class Broken(FakeLLM):
        def complete(self, messages, **kw):
            raise LLMError("quota")

    svc = make(sample_dir, Broken())
    assert svc.standalone_query("and 2023?", [{"role": "user", "content": "x"}]) == "and 2023?"
