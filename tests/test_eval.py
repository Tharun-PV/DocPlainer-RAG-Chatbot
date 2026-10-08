from pathlib import Path

from docplainer.config import Settings
from docplainer.eval.dataset import Gold, is_relevant, load_golden
from docplainer.eval.runner import build_baseline, build_v2, evaluate, summarise, sweep
from docplainer.schemas import Chunk

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "eval" / "golden.jsonl"


def chunk(text, file="a.pdf", page=1):
    return Chunk("c", "d", file, page, "", "text", text, text)


def test_relevance_matching_is_whitespace_and_case_insensitive():
    gold = [Gold("a.pdf", 1, "38 percent")]
    assert is_relevant(chunk("a 38percent drop"), gold)
    assert not is_relevant(chunk("a 38 percent drop", page=2), gold)
    assert not is_relevant(chunk("a 38 percent drop", file="b.pdf"), gold)
    assert not is_relevant(chunk("nothing"), gold)


def test_golden_set_is_well_formed():
    qs = load_golden(GOLDEN)
    assert len(qs) >= 25 and len({q.id for q in qs}) == len(qs)
    assert {q.type for q in qs} == {"text", "table", "scan", "unanswerable"}
    assert all(q.gold for q in qs if q.type != "unanswerable")


def test_v2_beats_baseline_on_sample_corpus(sample_dir):
    cfg = Settings(embedder_backend="hashing", use_reranker=False, min_dense_cosine=0.2)
    qs = load_golden(GOLDEN)
    v1 = summarise(evaluate(build_baseline(sample_dir, cfg), qs))
    v2 = summarise(evaluate(build_v2(sample_dir, cfg), qs))
    assert v1["scan"]["hit@k"] == 0.0  # pypdf cannot read scans
    assert v2["scan"]["hit@k"] >= 0.8
    assert v2["answerable (all)"]["hit@k"] >= 0.9
    assert v2["answerable (all)"]["mrr"] > v1["answerable (all)"]["mrr"]


def test_sweep_trades_answering_for_abstaining(sample_dir):
    cfg = Settings(embedder_backend="hashing", use_reranker=False)
    results = evaluate(build_v2(sample_dir, cfg), load_golden(GOLDEN))
    rows = sweep(results, [0.0, 0.99])
    assert rows[0]["abstained_ok"] == 0.0 and rows[1]["abstained_ok"] == 1.0
    assert rows[0]["answered_ok"] > rows[1]["answered_ok"]
