from collections import defaultdict

def read_trec_run(path):
    """
    Returns:
      dict[qid][docid] = rank
    """
    runs = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rank, _, _ = line.strip().split()
            runs[qid][docid] = int(rank)
    return runs


def weighted_rrf(
    run_files,      # list of (path, weight)
    output_file,
    k=60,
    run_tag="wrrf"
):
    """
    run_files: [(path, weight), ...]
    """
    run_dicts = [(read_trec_run(p), w) for p, w in run_files]

    fused = defaultdict(lambda: defaultdict(float))

    for run, weight in run_dicts:
        for qid, docs in run.items():
            for docid, rank in docs.items():
                fused[qid][docid] += weight / (k + rank)

    # write TREC run
    with open(output_file, "w", encoding="utf-8") as f:
        for qid in fused:
            ranked = sorted(
                fused[qid].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for i, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {i} {score:.6f} {run_tag}\n")
