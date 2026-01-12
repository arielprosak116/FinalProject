# eval_recall.py
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Set, List, Tuple


def load_qrels(qrels_path: str, rel_threshold: int = 1) -> Dict[str, Set[str]]:
    """
    qrels format: qid 0 docid rel
    """
    rel = defaultdict(set)
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, r = parts[0], parts[2], int(parts[3])
            if r >= rel_threshold:
                rel[qid].add(docid)
    return rel


def load_run(run_path: str) -> Dict[str, List[str]]:
    """
    TREC run: qid Q0 docid rank score tag
    """
    run = defaultdict(list)
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, docid = parts[0], parts[2]
            run[qid].append(docid)
    return run


def recall_at_k(qrels_path: str, run_path: str, k: int = 1000, rel_threshold: int = 1) -> float:
    rel = load_qrels(qrels_path, rel_threshold=rel_threshold)
    run = load_run(run_path)

    qids = sorted(set(rel.keys()) & set(run.keys()))
    if not qids:
        raise ValueError("No overlapping qids between qrels and run.")

    recalls = []
    for qid in qids:
        gold = rel[qid]
        if not gold:
            continue
        got = set(run[qid][:k])
        recalls.append(len(got & gold) / len(gold))

    return sum(recalls) / len(recalls) if recalls else 0.0
