from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"\w+(?:[.,]\w+)*", re.UNICODE)
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were what when where which who how with do does did".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase tokens that keep numbers intact ('26.2', '1,200') and fold simple plurals."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and not tok[-2].isdigit():
            tok = tok[:-1]
        out.append(tok)
    return out


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._df: Counter[str] = Counter()

    def fit(self, docs: list[str]) -> None:
        self._tf = [Counter(tokenize(d)) for d in docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._df = Counter(t for tf in self._tf for t in tf)

    def scores(self, query: str) -> list[float]:
        n = len(self._tf)
        if not n:
            return []
        avg = (sum(self._len) / n) or 1.0
        out = [0.0] * n
        for term in set(tokenize(query)):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._tf):
                f = tf.get(term, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self._len[i] / avg)
                    out[i] += idf * f * (self.k1 + 1) / norm
        return out
