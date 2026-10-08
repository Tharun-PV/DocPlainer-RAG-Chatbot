from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from docplainer.config import Settings
from docplainer.eval.dataset import Question, is_relevant, load_golden
from docplainer.ingestion import SUPPORTED_EXTS, ingest_files
from docplainer.retrieval import HybridIndex, Retriever, get_embedder, get_reranker
from docplainer.retrieval.retriever import RetrievalResult

Retrieve = Callable[[str], RetrievalResult]


@dataclass
class QuestionResult:
    id: str
    type: str
    answerable: bool
    first_rank: int | None  # 1-based rank of first relevant chunk, None if none retrieved
    confident: bool
    top_relevance: float


def build_v2(corpus: Path, cfg: Settings) -> Retrieve:
    files = sorted(p for p in corpus.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    result = ingest_files(files, cfg)
    for name, reason in result.errors.items():
        print(f"  ! could not ingest {name}: {reason}")
    index = HybridIndex(get_embedder(cfg))
    index.add(result.chunks)
    return Retriever(index, get_reranker(cfg), cfg).retrieve


def build_baseline(corpus: Path, cfg: Settings) -> Retrieve:
    from docplainer.eval.baseline import BaselineRetriever

    files = sorted(corpus.glob("*.pdf"))
    return BaselineRetriever(files, get_embedder(cfg)).retrieve


def evaluate(retrieve: Retrieve, questions: list[Question]) -> list[QuestionResult]:
    out = []
    for q in questions:
        res = retrieve(q.question)
        rank = next(
            (i for i, h in enumerate(res.hits, start=1) if is_relevant(h.chunk, q.gold)), None
        )
        top = max((h.relevance for h in res.hits), default=0.0)
        out.append(QuestionResult(q.id, q.type, q.answerable, rank, res.confident, top))
    return out


def summarise(results: list[QuestionResult]) -> dict[str, dict[str, float]]:
    """Per-type and overall metrics. hit@k / MRR on answerable questions only; abstain on rest."""
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        groups[r.type].append(r)
        if r.answerable:
            groups["answerable (all)"].append(r)
    table: dict[str, dict[str, float]] = {}
    for name, rs in groups.items():
        if rs[0].answerable:
            n = len(rs)
            table[name] = {
                "n": n,
                "hit@1": sum(1 for r in rs if r.first_rank == 1) / n,
                "hit@3": sum(1 for r in rs if r.first_rank and r.first_rank <= 3) / n,
                "hit@k": sum(1 for r in rs if r.first_rank) / n,
                "mrr": statistics.fmean(1 / r.first_rank if r.first_rank else 0 for r in rs),
            }
        else:
            table[name] = {
                "n": len(rs),
                "abstain": sum(1 for r in rs if not r.confident) / len(rs),
            }
    return table


def sweep(results: list[QuestionResult], thresholds: list[float]) -> list[dict[str, float]]:
    """How the relevance gate trades answering correctly against refusing out-of-scope questions."""
    rows = []
    ans = [r for r in results if r.answerable]
    una = [r for r in results if not r.answerable]
    for t in thresholds:
        rows.append(
            {
                "threshold": t,
                "answered_ok": sum(1 for r in ans if r.first_rank and r.top_relevance >= t)
                / max(len(ans), 1),
                "abstained_ok": sum(1 for r in una if r.top_relevance < t) / max(len(una), 1),
            }
        )
    return rows


def format_table(name: str, summary: dict[str, dict[str, float]]) -> str:
    head = f"  {'slice':<18}{'n':>3}  {'hit@1':>6} {'hit@3':>6} {'hit@k':>6} {'mrr':>6}"
    lines = [f"\n{name}", head]
    for key, m in summary.items():
        if "hit@1" in m:
            lines.append(
                f"  {key:<18}{int(m['n']):>3}  {m['hit@1']:>6.2f} {m['hit@3']:>6.2f} "
                f"{m['hit@k']:>6.2f} {m['mrr']:>6.2f}"
            )
        else:
            lines.append(f"  {key:<18}{int(m['n']):>3}  correctly abstained: {m['abstain']:.2f}")
    return "\n".join(lines)


def run(
    corpus: Path,
    golden: Path,
    cfg: Settings,
    compare_baseline: bool = True,
    out_json: Path | None = None,
) -> dict:
    questions = load_golden(golden)
    report: dict = {"embedder": cfg.embedder_backend, "n_questions": len(questions)}
    systems = {"v2 (new)": build_v2}
    if compare_baseline:
        systems = {"v1 baseline": build_baseline, **systems}
    for name, builder in systems.items():
        results = evaluate(builder(corpus, cfg), questions)
        summary = summarise(results)
        print(format_table(name, summary))
        report[name] = {"summary": summary, "results": [asdict(r) for r in results]}
    if out_json:
        out_json.write_text(json.dumps(report, indent=2))
    return report
