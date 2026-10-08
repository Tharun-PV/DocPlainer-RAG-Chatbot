from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docplainer.config import Settings
from docplainer.eval.dataset import load_golden
from docplainer.eval.runner import build_v2, evaluate, run, sweep


def main() -> int:
    ap = argparse.ArgumentParser(description="Score retrieval quality on a golden question set.")
    ap.add_argument("--corpus", type=Path, default=Path("data/sample"))
    ap.add_argument("--golden", type=Path, default=Path("data/eval/golden.jsonl"))
    ap.add_argument("--embedder", choices=["fastembed", "hashing"], default=None)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--json", type=Path, default=None, help="write full results here")
    ap.add_argument("--sweep", action="store_true", help="print relevance-threshold trade-off")
    ap.add_argument("--min-hit-at-k", type=float, default=None, help="exit 1 if v2 falls below")
    args = ap.parse_args()

    overrides: dict = {}
    if args.embedder:
        overrides["embedder_backend"] = args.embedder
    if args.no_rerank:
        overrides["use_reranker"] = False
    cfg = Settings(**overrides)
    print(f"embedder={cfg.embedder_backend} reranker={cfg.use_reranker}")

    report = run(args.corpus, args.golden, cfg, not args.no_baseline, args.json)

    if args.sweep:
        results = evaluate(build_v2(args.corpus, cfg), load_golden(args.golden))
        print("\nrelevance gate sweep (v2)\n  threshold  answered_ok  abstained_ok")
        for row in sweep(results, [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
            print(
                f"  {row['threshold']:>9.2f}  {row['answered_ok']:>11.2f}  "
                f"{row['abstained_ok']:>12.2f}"
            )

    if args.min_hit_at_k is not None:
        got = report["v2 (new)"]["summary"]["answerable (all)"]["hit@k"]
        if got < args.min_hit_at_k:
            print(f"\nFAIL: v2 hit@k {got:.2f} < required {args.min_hit_at_k:.2f}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
