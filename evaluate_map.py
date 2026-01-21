from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional
from collections import defaultdict
import pandas as pd


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

def get_map_by_paths(qrels_path, run_path):
    qrels = load_qrels(qrels_path)  # or "qrel301.txt"
    run = load_run(run_path)

    map_score, ap_by_q = mean_average_precision(qrels, run)
    return map_score

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


def precision_at_k(ranked_docids, qrels_for_q, k):
    rel_set = {d for d, rel in qrels_for_q.items() if rel > 0}
    if not ranked_docids:
        return 0.0
    return sum(1 for d in ranked_docids[:k] if d in rel_set) / k


def recall_at_k(ranked_docids, qrels_for_q, k):
    rel_set = {d for d, rel in qrels_for_q.items() if rel > 0}
    if not rel_set:
        return 0.0
    return sum(1 for d in ranked_docids[:k] if d in rel_set) / len(rel_set)


def max_ap_at_k(qrels_for_q: Dict[str, int], k: int) -> float:
    rel_count = sum(1 for _, rel in qrels_for_q.items() if rel > 0)
    if rel_count == 0:
        return 0.0
    return min(rel_count, k) / rel_count


def first_relevant_rank_at_k(ranked_docids: List[str], qrels_for_q: Dict[str, int], k: int) -> int:
    rel_set = {d for d, rel in qrels_for_q.items() if rel > 0}
    if not rel_set:
        return 0  # no relevant docs judged for this query
    for i, docid in enumerate(ranked_docids[:k], start=1):
        if docid in rel_set:
            return i
    return 0  # none found within top-k


def reciprocal_rank_at_k(ranked_docids: List[str], qrels_for_q: Dict[str, int], k: int) -> float:
    r = first_relevant_rank_at_k(ranked_docids, qrels_for_q, k)
    return 1.0 / r if r > 0 else 0.0


def fbeta(p: float, r: float, beta: float = 2.0) -> float:
    if p <= 0.0 or r <= 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * p * r / (b2 * p + r)


def macro_fbeta_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, List[str]],
    k: int,
    beta: float = 2.0
) -> float:
    vals = []
    for qid, qrels_for_q in qrels.items():
        ranked = run.get(qid, [])
        p = precision_at_k(ranked, qrels_for_q, k)
        r = recall_at_k(ranked, qrels_for_q, k)
        vals.append(fbeta(p, r, beta=beta))
    return sum(vals) / len(vals)


def evaluate_run(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, List[str]],
    ks=range(100, 1001, 100),
) -> pd.DataFrame:
    """
    Columns: k, MAP, P, Recall, MaxAP, FirstRel, MRR
    """
    rows = []

    for k in ks:
        ap_vals, p_vals, r_vals, max_ap_vals = [], [], [], []
        first_vals, rr_vals = [], []

        for qid, qrels_for_q in qrels.items():
            ranked = run.get(qid, [])[:k]

            ap_vals.append(average_precision(ranked, qrels_for_q))
            p_vals.append(precision_at_k(ranked, qrels_for_q, k))
            r_vals.append(recall_at_k(ranked, qrels_for_q, k))
            max_ap_vals.append(max_ap_at_k(qrels_for_q, k))

            fr = first_relevant_rank_at_k(ranked, qrels_for_q, k)
            first_vals.append(fr)
            rr_vals.append(1.0 / fr if fr > 0 else 0.0)

        rows.append({
            "k": int(k),
            "MAP": sum(ap_vals) / len(ap_vals),
            "P": sum(p_vals) / len(p_vals),
            "Recall": sum(r_vals) / len(r_vals),
            "MaxAP": sum(max_ap_vals) / len(max_ap_vals),
            "FirstRel": sum(first_vals) / len(first_vals),  # mean first relevant rank (0 if none)
            "MRR": sum(rr_vals) / len(rr_vals),            # mean reciprocal rank@k
        })

    df = pd.DataFrame(rows)
    return df

def missed_relevant_at_k(ranked_docids: List[str], qrels_for_q: Dict[str, int], k: int) -> int:
    """
    # of judged relevant docs NOT retrieved in top-k.
    """
    rel_set = {d for d, rel in qrels_for_q.items() if rel > 0}
    if not rel_set:
        return 0
    retrieved_rel = {d for d in ranked_docids[:k] if d in rel_set}
    return len(rel_set - retrieved_rel)


def hardest_queries_report(
    qrels: Dict[str, Dict[str, int]],
    run_before: Dict[str, List[str]],
    run_after: Optional[Dict[str, List[str]]] = None,
    k: int = 1000,
    top_x: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Returns top-x hardest queries under multiple categories at cutoff k.

    Categories:
      - lowest_recall: lowest Recall@k (before)
      - most_missed: most missed relevant docs in top-k (before)
      - largest_ap_drop: most negative AP delta (after - before)  [if run_after provided]
      - smallest_ap_gain: smallest AP delta (after - before)      [if run_after provided]

    Requires these existing helpers (defined earlier in your codebase):
      - average_precision(ranked_docids, qrels_for_q)
      - precision_at_k(ranked_docids, qrels_for_q, k)
      - recall_at_k(ranked_docids, qrels_for_q, k)
    """
    rows = []

    for qid, qrels_for_q in qrels.items():
        ranked_b = run_before.get(qid, [])[:k]

        row = {
            "qid": qid,
            "num_rel": sum(1 for _, rel in qrels_for_q.items() if rel > 0),
            "ap_before": average_precision(ranked_b, qrels_for_q),
            "p_before": precision_at_k(ranked_b, qrels_for_q, k),
            "recall_before": recall_at_k(ranked_b, qrels_for_q, k),
            "missed_before": missed_relevant_at_k(ranked_b, qrels_for_q, k),
        }

        if run_after is not None:
            ranked_a = run_after.get(qid, [])[:k]
            ap_a = average_precision(ranked_a, qrels_for_q)

            row.update({
                "ap_after": ap_a,
                "p_after": precision_at_k(ranked_a, qrels_for_q, k),
                "recall_after": recall_at_k(ranked_a, qrels_for_q, k),
                "missed_after": missed_relevant_at_k(ranked_a, qrels_for_q, k),
                "ap_delta": ap_a - row["ap_before"],
            })

        rows.append(row)

    df = pd.DataFrame(rows)

    out: Dict[str, pd.DataFrame] = {"lowest_recall": (
        df.sort_values(["recall_before", "missed_before", "num_rel"], ascending=[True, False, False])
        .head(top_x)
        .reset_index(drop=True)
    ), "most_missed": (
        df.sort_values(["missed_before", "num_rel", "recall_before"], ascending=[False, False, True])
        .head(top_x)
        .reset_index(drop=True)
    )}

    if run_after is not None:
        out["largest_ap_drop"] = (
            df.sort_values(["ap_delta", "missed_after", "missed_before"], ascending=[True, False, False])
              .head(top_x)
              .reset_index(drop=True)
        )

        out["smallest_ap_gain"] = (
            df.sort_values(["ap_delta", "recall_before", "missed_before"], ascending=[True, True, False])
              .head(top_x)
              .reset_index(drop=True)
        )

    return out


