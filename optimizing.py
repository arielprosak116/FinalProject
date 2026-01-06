from engine import SearchEngine
from evaluate_map import *


class Optimize:
    def __init__(self):
        self.se = SearchEngine()

    def optimize_qld(self,
                     topics,
                     fb_terms_values,
                     fb_docs_values,
                     og_query_weight_values,
                     k=200,
                     qrels_path="Data/qrels_50_Queries",
                     results_file="qld_results.txt"):

        with open(results_file, "w", encoding="utf-8") as f:
            f.write("fb_terms\tfb_docs\toriginal_query_weight\tMAP\n")

        best_map = -1.0
        best_params = None
        qrels = load_qrels(qrels_path)  # or "qrel301.txt"

        for fb_terms in fb_terms_values:
            for fb_docs in fb_docs_values:
                for original_query_weight in og_query_weight_values:
                    self.se.set_searcher(approach="qld",
                                         fb_terms=fb_terms,
                                         fb_docs=fb_docs,
                                         original_query_weight=original_query_weight)

                    self.se.search_all_queries(topics, k)
                    run = load_run("run.txt")

                    map_score, ap_by_q = mean_average_precision(qrels, run)

                    with open(results_file, "a", encoding="utf-8") as f:
                        f.write(f"{fb_terms}\t{fb_docs}\t{original_query_weight}\t{map_score:.6f}\n")
                    print(f"fb_terms={fb_terms}, fb_docs={fb_docs}, "
                        f"w={original_query_weight} -> MAP={map_score:.6f}")

                    if map_score > best_map:
                        best_map = map_score
                        best_params = (fb_terms, fb_docs, original_query_weight)

                    # print(fb_term, fb_docs, original_query_weight + " map is " + map_score)

        print("\nBEST:")
        print(f"fb_terms={best_params[0]}, fb_docs={best_params[1]}, "
              f"w={best_params[2]} -> MAP={best_map:.6f}")