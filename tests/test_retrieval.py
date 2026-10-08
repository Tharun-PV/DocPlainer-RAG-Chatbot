from docplainer.config import Settings
from docplainer.ingestion import ingest_files
from docplainer.retrieval import HybridIndex, Retriever
from docplainer.retrieval.bm25 import BM25, tokenize
from docplainer.retrieval.embeddings import HashingEmbedder
from docplainer.schemas import Chunk


def mk(i: int, text: str, doc="d1", page=1) -> Chunk:
    return Chunk(f"{doc}:{i}", doc, f"{doc}.pdf", page, "", "text", text, text)


def test_tokenize_keeps_numbers_and_folds_plurals():
    assert tokenize("Revenue grew 26.2% to 1,200 reports") == [
        "revenue",
        "grew",
        "26.2",
        "1,200",
        "report",
    ]


def test_bm25_prefers_rare_exact_terms():
    bm = BM25()
    bm.fit(["the quick brown fox", "invoice INV-2291 is overdue", "brown bread recipe"])
    s = bm.scores("INV-2291")
    assert s.index(max(s)) == 1 and s[0] == 0


def test_index_dedupes_documents_and_supports_removal_and_filters():
    idx = HybridIndex(HashingEmbedder())
    a = [mk(1, "alpha beta gamma"), mk(2, "delta epsilon")]
    b = [mk(1, "alpha zeta", doc="d2")]
    assert idx.add(a) == 2 and idx.add(a) == 0 and idx.add(b) == 1
    assert {h.chunk.doc_id for h in idx.search("alpha")} == {"d1", "d2"}
    assert {h.chunk.doc_id for h in idx.search("alpha", doc_ids={"d2"})} == {"d2"}
    idx.remove_doc("d1")
    assert len(idx) == 1
    assert {h.chunk.doc_id for h in idx.search("delta zeta")} == {"d2"}


def test_rrf_rewards_chunks_found_by_both_retrievers():
    idx = HybridIndex(HashingEmbedder())
    idx.add([mk(1, "cats sit on mats"), mk(2, "code INV-77 refers to invoice"), mk(3, "dogs bark")])
    hits = idx.search("invoice INV-77")
    assert hits[0].chunk.id == "d1:2" and hits[0].dense_rank and hits[0].sparse_rank


class FakeReranker:
    def __init__(self, favourite: str):
        self.favourite = favourite

    def score(self, query, texts):
        return [8.0 if self.favourite in t else -9.0 for t in texts]


def test_reranker_reorders_and_gate_uses_its_score(sample_dir):
    cfg = Settings(embedder_backend="hashing")
    idx = HybridIndex(HashingEmbedder())
    idx.add(ingest_files([sample_dir / "northwind_annual_report.pdf"], cfg).chunks)

    good = Retriever(idx, FakeReranker("141.9"), cfg).retrieve("europe revenue")
    assert good.confident and "141.9" in good.hits[0].chunk.text

    bad = Retriever(idx, FakeReranker("nothing matches this"), cfg).retrieve("europe revenue")
    assert not bad.confident and bad.trace["reranked"]


def test_empty_index_is_not_confident():
    res = Retriever(HybridIndex(HashingEmbedder()), None, Settings()).retrieve("anything")
    assert res.hits == [] and not res.confident
