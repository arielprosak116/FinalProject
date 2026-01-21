from engine import SearchEngine
from evaluate_map import *
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Tuple, Any
import pandas as pd
import json
import json
checkpoint_path = Path("Results/tuning/checkpoint_metrics.jsonl")



class Optimize:
    def __init__(self):
        self.se = SearchEngine()

    def optimize_qld(self,
                     topics,
                     fb_terms_values,
                     fb_docs_values,
                     og_query_weight_values,
                     mus,
                     k=200,
                     qrels_path="Data/qrels_50_Queries",
                     results_file="qld_mu_results.txt"):

        with open(results_file, "w", encoding="utf-8") as f:
            f.write("fb_terms\tfb_docs\toriginal_query_weight\tMAP\n")

        best_map = -1.0
        best_params = None
        qrels = load_qrels(qrels_path)  # or "qrel301.txt"

        for fb_terms in fb_terms_values:
            for fb_docs in fb_docs_values:
                for original_query_weight in og_query_weight_values:
                    for mu in mus:
                        self.se.set_searcher(approach="qld",
                                             fb_terms=fb_terms,
                                             fb_docs=fb_docs,
                                             original_query_weight=original_query_weight,
                                             mu=mu)

                        self.se.search_all_queries(topics, k)
                        run = load_run("Results/run.txt")

                        map_score, ap_by_q = mean_average_precision(qrels, run)

                        with open(results_file, "a", encoding="utf-8") as f:
                            f.write(f"{fb_terms}\t{fb_docs}\t{original_query_weight}\t{mu}\t{map_score:.6f}\n")
                        print(f"fb_terms={fb_terms}, fb_docs={fb_docs}, "
                            f"w={original_query_weight}, mu={mu} -> MAP={map_score:.6f}")

                        if map_score > best_map:
                            best_map = map_score
                            best_params = (fb_terms, fb_docs, original_query_weight, mu)

                        # print(fb_term, fb_docs, original_query_weight + " map is " + map_score)

        print("\nBEST:")
        print(f"fb_terms={best_params[0]}, fb_docs={best_params[1]}, "
              f"w={best_params[2]}, mu={best_params[3]} -> MAP={best_map:.6f}")

    def tune_bm25_rm3_rrf(self,
                          qrels: Dict[str, Dict[str, int]],
                          topics_lists: List[Dict[str, str]],
                          query_fusion_weights_lists: List[List[float]],
                          fb_terms_values: Iterable[int],
                          fb_docs_values: Iterable[int],
                          bm25_k1_values: Iterable[float] = (0.5, 0.9, 1.2, 1.5, 1.8),
                          bm25_b_values: Iterable[float] = (0.3, 0.36, 0.5, 0.7, 0.9),
                          rrf_k_values: Iterable[int] = (10, 20, 60),
                          eval_recall_k: int = 1000,
                          out_dir: str | Path = "Results/tuning",
                          run_tag: str = "bm25rm3_tune",
                          ) -> Dict[str, pd.DataFrame]:
        """
        Grid-search tuning for a BM25+RM3 retrieval stage that fuses multiple query variants via RRF.
        Returns:
          (all_results_df, best_by_map_df, best_by_recall_df)

          - all_results_df: one row per parameter setting with MAP@eval_recall_k and Recall@eval_recall_k + P@100.
          - best_by_map_df: single-row df of the best configuration by MAP@eval_recall_k.
          - best_by_recall_df: single-row df of the best configuration by Recall@eval_recall_k (tie-break MAP).
        """

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows: List[dict] = []

        def map_at_k(run: Dict[str, List[str]], k: int) -> float:
            vals = []
            for qid, qrels_for_q in qrels.items():
                ranked = run.get(qid, [])[:k]
                vals.append(average_precision(ranked, qrels_for_q))
            return sum(vals) / len(vals)

        def mean_recall_at_k(run: Dict[str, List[str]], k: int) -> float:
            vals = []
            for qid, qrels_for_q in qrels.items():
                ranked = run.get(qid, [])
                vals.append(recall_at_k(ranked, qrels_for_q, k))
            return sum(vals) / len(vals)

        def mean_precision_at_k(run: Dict[str, List[str]], k: int) -> float:
            vals = []
            for qid, qrels_for_q in qrels.items():
                ranked = run.get(qid, [])
                vals.append(precision_at_k(ranked, qrels_for_q, k))
            return sum(vals) / len(vals)

        # --- grid search ---
        all_perms = len(query_fusion_weights_lists)*len(fb_terms_values)*len(fb_docs_values)*len(bm25_k1_values)*len(bm25_b_values)*len(rrf_k_values)
        i=0
        for query_fusion_weights in query_fusion_weights_lists:
            for fb_terms in fb_terms_values:
                for fb_docs in fb_docs_values:
                    for k1 in bm25_k1_values:
                        for b in bm25_b_values:
                            for rrf_k in rrf_k_values:
                                self.se.set_searcher(approach="bm25", k1=k1, b=b, fb_terms=fb_terms, reranker_type=None)
                                run_name = f"{run_tag}_fbT{fb_terms}_fbD{fb_docs}_k1{k1}_b{b}_rrf{rrf_k}"
                                run_path = out_dir / run_name
                                self.se.search_all_queries(
                                    topics_lists=topics_lists,
                                    llm_query_fusion_weights=query_fusion_weights,
                                    rrf_k_queries=rrf_k,
                                    output_dir=out_dir,
                                    output_file=str(run_name),
                                )

                                run = load_run(f"{str(run_path)}_rrf_rerank_1.txt")

                                # Evaluate (simple set: MAP + Recall at chosen k; plus a fixed early precision)
                                map_k = map_at_k(run, eval_recall_k)
                                recall_k = mean_recall_at_k(run, eval_recall_k)
                                p100 = mean_precision_at_k(run, 100)
                                f2_macro = macro_fbeta_at_k(qrels, run, eval_recall_k, beta=2.0)
                                row = {"query_fusion_weights": query_fusion_weights,
                                    "fb_terms": int(fb_terms),
                                    "fb_docs": int(fb_docs),
                                    "bm25_k1": float(k1),
                                    "bm25_b": float(b),
                                    "rrf_k": int(rrf_k),
                                    f"MAP@{eval_recall_k}": map_k,
                                    f"Recall@{eval_recall_k}": recall_k,
                                    f"F2_macro@{eval_recall_k}": f2_macro,
                                    "P@100": p100,
                                    "run_path": str(run_path)}
                                rows.append(row)
                                with checkpoint_path.open("a", encoding="utf-8") as f:
                                    f.write(json.dumps(row) + "\n")
                                i+=1
                                print(f"{i}/{all_perms} Done")

        all_df = pd.DataFrame(rows).sort_values([f"MAP@{eval_recall_k}"], ascending=False).reset_index(drop=True)

        # Best by MAP (primary), tie-break by Recall
        best_by_map = (
            all_df.sort_values([f"MAP@{eval_recall_k}", f"Recall@{eval_recall_k}", "P@100"],
                               ascending=[False, False, False])
            .head(1)
            .reset_index(drop=True)
        )

        # Best by Recall (primary), tie-break by MAP
        best_by_recall = (
            all_df.sort_values([f"Recall@{eval_recall_k}", f"MAP@{eval_recall_k}", "P@100"],
                               ascending=[False, False, False])
            .head(1)
            .reset_index(drop=True)
        )

        # Best by F2 (primary), tie-break by MAP
        best_by_f2 = (
            all_df.sort_values([f"F2_macro@{eval_recall_k}", f"MAP@{eval_recall_k}", "P@100"],
                               ascending=[False, False, False])
            .head(1)
            .reset_index(drop=True)
        )

        # Optional: also report the best "balanced" configuration by a simple combined score
        all_df["combo_score"] = all_df[f"MAP@{eval_recall_k}"] + all_df[f"Recall@{eval_recall_k}"]
        best_combo = (
            all_df.sort_values(["combo_score", f"MAP@{eval_recall_k}", f"Recall@{eval_recall_k}"],
                               ascending=[False, False, False])
            .head(1)
            .reset_index(drop=True)
        )

        return {'overall': all_df, 'MAP': best_by_map, 'Recall': best_by_recall, 'F2': best_by_f2}
