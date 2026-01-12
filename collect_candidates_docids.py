# collect_candidate_docids.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Optional
from pyserini.search.lucene import LuceneSearcher


def load_queries_robust(path: str) -> Dict[str, str]:
    """
    Supports:
      qid<TAB>query
      qid: query
      qid query
    """
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                qid, q = line.split("\t", 1)
            elif ":" in line[:12]:
                qid, q = line.split(":", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                qid, q = parts
            queries[qid.strip()] = q.strip()
    return queries


def collect_candidate_docids(
    searcher,
    queries_path: str,
    out_path: str = "candidate_docids.txt",
    index_name: str = "robust04",
    approach: str = "qld",   # "qld" or "bm25"
    mu: float = 340,
    rm3: bool = True,
    fb_terms: int = 20,
    fb_docs: int = 5,
    original_query_weight: float = 0.6,
    topk_per_query: int = 200,
    max_queries: Optional[int] = None,
    progress_every: int = 25,
) -> str:
    """
    Runs retrieval for all queries and stores unique docids from topk_per_query results.
    """
    if approach == "qld":
        searcher.set_qld(mu=mu)
        if rm3:
            searcher.set_rm3(
                fb_terms=fb_terms,
                fb_docs=fb_docs,
                original_query_weight=original_query_weight
            )
    elif approach == "bm25":
        searcher.set_bm25(k1=0.9, b=0.4)
    else:
        raise ValueError("approach must be 'qld' or 'bm25'")

    queries = load_queries_robust(queries_path)
    qids = list(queries.keys())
    if max_queries is not None:
        qids = qids[:max_queries]

    docids: Set[str] = set()

    for i, qid in enumerate(qids, start=1):
        hits = searcher.search(queries[qid], k=topk_per_query)
        for h in hits:
            docids.add(h.docid)

        if progress_every and i % progress_every == 0:
            print(f"[collect_docids] {i}/{len(qids)} queries, unique docids={len(docids)}")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for d in sorted(docids):
            f.write(d + "\n")

    print(f"[collect_docids] wrote {len(docids)} docids to {out_file}")
    return str(out_file)
