# rrf_fuse.py
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple


def read_trec_run(path: str) -> Dict[str, List[str]]:
    """
    Returns qid -> ranked list of docids (in order).
    """
    q = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank = parts[0], parts[1], parts[2], int(parts[3])
            q[qid].append(docid)
    return q


def write_trec_run(path: str, scores: Dict[str, Dict[str, float]], tag: str, k: int = 1000):
    with open(path, "w", encoding="utf-8") as f:
        for qid in scores:
            ranked = sorted(scores[qid].items(), key=lambda x: x[1], reverse=True)[:k]
            for r, (docid, sc) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {r} {sc:.6f} {tag}\n")


def rrf_fuse(run_a: str, run_b: str, out_path: str, k: int = 1000, rrf_k: int = 60, tag: str = "rrf"):
    """
    RRF score(doc) = sum_runs 1 / (rrf_k + rank)
    """
    A = read_trec_run(run_a)
    B = read_trec_run(run_b)

    all_qids = set(A.keys()) | set(B.keys())
    fused = {qid: defaultdict(float) for qid in all_qids}

    def add_run(run: Dict[str, List[str]]):
        for qid, docs in run.items():
            for rank, docid in enumerate(docs[:k], start=1):
                fused[qid][docid] += 1.0 / (rrf_k + rank)

    add_run(A)
    add_run(B)

    write_trec_run(out_path, fused, tag=tag, k=k)
    return out_path
