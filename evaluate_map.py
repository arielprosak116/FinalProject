from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional

def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """
    qrels line format (TREC):
      qid  unused  docid  rel
    Example:
      301  0       FBIS3-10555  0
    """
    qrels = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, _unused, docid, rel = line.split()[:4]
            qrels[qid][docid] = int(rel)
    return qrels

def load_run(run_path: str) -> Dict[str, List[str]]:
    """
    run line format (TREC run file):
      qid  Q0  docid  rank  score  tag
    We will sort by rank (int) to be safe.
    """
    run = defaultdict(list)  # qid -> list[(rank, docid)]
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            qid, _q0, docid, rank = parts[0], parts[1], parts[2], parts[3]
            run[qid].append((int(rank), docid))
    # sort by rank then keep docids
    out = {}
    for qid, lst in run.items():
        lst.sort(key=lambda x: x[0])
        out[qid] = [docid for _, docid in lst]
    return out

def average_precision(ranked_docids: List[str], qrels_for_q: Dict[str, int]) -> float:
    """
    AP(q) = average over precisions at ranks where a relevant document is found.
    Relevant is rel > 0. Unjudged docs are treated as non-relevant.
    Denominator is #relevant judged docs for that query.
    """
    rel_set = {docid for docid, rel in qrels_for_q.items() if rel > 0}
    if not rel_set:
        return 0.0

    hits = 0
    sum_prec = 0.0
    for i, docid in enumerate(ranked_docids, start=1):
        if docid in rel_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(rel_set)

def mean_average_precision(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, List[str]],
    qids: Optional[Iterable[str]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (MAP, per_query_AP_dict).
    If qids is None: evaluate intersection of qrels and run query ids.
    """
    if qids is None:
        eval_qids = sorted(set(qrels.keys()) & set(run.keys()), key=lambda x: int(x) if x.isdigit() else x)
    else:
        eval_qids = [str(q) for q in qids]

    ap_by_q = {}
    ap_values = []
    for qid in eval_qids:
        ap = average_precision(run.get(qid, []), qrels.get(qid, {}))
        ap_by_q[qid] = ap
        ap_values.append(ap)

    map_score = sum(ap_values) / len(ap_values) if ap_values else 0.0
    return map_score, ap_by_q

def load_topics(path):
    """
    Input format:
    qid<TAB>query text
    """
    topics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, query = line.split("\t", 1)
            topics[qid] = query
    return topics
