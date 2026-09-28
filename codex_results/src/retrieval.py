"""Scalable sparse retrieval primitives.

* Hashed features (no vocabulary dict) -> bounded memory for millions of docs.
* IDF from the target corpus; very frequent features are dropped (max_df).
* Query batches x target matrix via scipy sparse matmul; vectorized top-k.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

N_FEAT = 2 ** 22


def make_hasher(kind: str) -> HashingVectorizer:
    common = dict(n_features=N_FEAT, alternate_sign=False, norm=None, lowercase=False, dtype=np.float32)
    if kind == "word12":
        return HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", **common)
    if kind == "word1":
        return HashingVectorizer(analyzer="word", ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b", **common)
    if kind == "ws":  # whitespace tokens, used for pre-built token strings (codes, fuzzy keys)
        return HashingVectorizer(analyzer=str.split, **common)
    if kind.startswith("charwb"):  # e.g. charwb34
        lo, hi = int(kind[-2]), int(kind[-1])
        return HashingVectorizer(analyzer="char_wb", ngram_range=(lo, hi), **common)
    if kind.startswith("char"):
        lo, hi = int(kind[-2]), int(kind[-1])
        return HashingVectorizer(analyzer="char", ngram_range=(lo, hi), **common)
    raise ValueError(kind)


class HashedTfidf:
    """TF-IDF over hashed features, fitted on one corpus (the targets)."""

    def __init__(self, kind: str, max_df: float = 0.02, sublinear: bool = True, binary: bool = False):
        self.kind, self.max_df, self.sublinear, self.binary = kind, max_df, sublinear, binary
        self.hasher = make_hasher(kind)
        self.idf = None

    def _raw(self, texts) -> sparse.csr_matrix:
        X = self.hasher.transform(texts).tocsr()
        X.sum_duplicates()
        if self.binary:
            X.data[:] = 1.0
        elif self.sublinear:
            np.log(X.data, out=X.data)
            X.data += 1.0
        return X

    def fit_transform(self, texts) -> sparse.csr_matrix:
        X = self._raw(texts)
        n = X.shape[0]
        df = np.bincount(X.indices, minlength=N_FEAT).astype(np.float32)
        idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        idf[df > max(self.max_df * n, 50)] = 0.0  # drop uninformative high-df features
        self.idf = idf.astype(np.float32)
        self.df = df
        return self._weight(X)

    def transform(self, texts) -> sparse.csr_matrix:
        return self._weight(self._raw(texts))

    def _weight(self, X):
        X.data *= self.idf[X.indices]
        X.eliminate_zeros()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        X = sparse.diags((1.0 / norms).astype(np.float32)) @ X
        return X.tocsr()


def topk_rows(M: sparse.csr_matrix, k: int):
    """Return (row, col, score, rank) arrays of the top-k entries per row."""
    M = M.tocsr()
    ip, data, ind = M.indptr, M.data, M.indices
    R, C, V, K = [], [], [], []
    for i in range(M.shape[0]):
        a, b = ip[i], ip[i + 1]
        if a == b:
            continue
        d = data[a:b]
        if b - a > k:
            p = np.argpartition(d, -k)[-k:]
            p = p[np.argsort(-d[p], kind="stable")]
        else:
            p = np.argsort(-d, kind="stable")
        n = len(p)
        R.append(np.full(n, i, np.int32)); C.append(ind[a:b][p]); V.append(d[p]); K.append(np.arange(n, dtype=np.int32))
    if not R:
        e = np.array([], np.int32)
        return e, e, np.array([], np.float32), e
    return np.concatenate(R), np.concatenate(C).astype(np.int32), np.concatenate(V), np.concatenate(K)


def search(Q: sparse.csr_matrix, TT: sparse.csr_matrix, k: int, batch: int = 1000, min_score: float = 0.0):
    """Top-k targets for every query row; TT is the transposed target matrix (features x docs, CSR).
    Returns arrays (q, t, score, rank)."""
    out = [[], [], [], []]
    for s in range(0, Q.shape[0], batch):
        S = (Q[s:s + batch] @ TT).tocsr()
        if min_score > 0:
            S.data[S.data < min_score] = 0
            S.eliminate_zeros()
        r, c, v, rk = topk_rows(S, k)
        out[0].append(r + s); out[1].append(c); out[2].append(v); out[3].append(rk)
        del S
    return tuple(np.concatenate(x) if x else np.array([]) for x in out)
