from engine import SearchEngine
from evaluate_map import *


class Optimize:
    def __init__(self):
        self.se = SearchEngine()

    def optimize_qld(self,
                     topics,
                     approach="qld",
                     fb_terms_values=None,
                     fb_docs_values=None,
                     og_query_weight_values=None,
                     mus=None,
                     k1s=None,
                     bs=None,
                     k=200,
                     qrels_path="Data/qrels_50_Queries",
                     results_file="bm25rm3_best_results.txt",
                     run_file_path="run.txt"):

        with open(results_file, "w", encoding="utf-8") as f:
            f.write("fb_terms\tfb_docs\toriginal_query_weight\tMAP\n")

        best_map = -1.0
        best_params = None
        qrels = load_qrels(qrels_path)  # or "qrel301.txt"

        # if k1s and bs:
        #     for k1 in k1s:
        #         for b in bs:
        #             self.se.set_searcher(approach="bm25", k1=k1, b=b)
        #             self.se.search_all_queries(topics, k)
        #             run = load_run("run.txt")

        for k1 in k1s:
            for b in bs:
                for fb_terms in fb_terms_values:
                    for fb_docs in fb_docs_values:
                        for original_query_weight in og_query_weight_values:
                            for mu in mus:
                                self.se.set_searcher(approach=approach,
                                                     fb_terms=fb_terms,
                                                     fb_docs=fb_docs,
                                                     original_query_weight=original_query_weight,
                                                     mu=mu,
                                                     k1=k1,
                                                     b=b)

                                self.se.search_all_queries(topics, k, output_file=run_file_path)
                                run = load_run(run_file_path)

                                map_score, ap_by_q = mean_average_precision(qrels, run)

                                with open(results_file, "a", encoding="utf-8") as f:
                                    f.write(f"{fb_terms}\t{fb_docs}\t{original_query_weight}\t"
                                            f"{mu}\t{k}\t{b}\t{map_score:.6f}\n")
                                print(f"fb_terms={fb_terms}, fb_docs={fb_docs}, "
                                    f"w={original_query_weight}, mu={mu}"
                                      f"k1={k1}, b={b} -> MAP={map_score:.6f}")

                                if map_score > best_map:
                                    best_map = map_score
                                    best_params = (fb_terms, fb_docs, original_query_weight, mu, k1, b)

                                # print(fb_term, fb_docs, original_query_weight + " map is " + map_score)

        print("\nBEST:")
        print(f"fb_terms={best_params[0]}, fb_docs={best_params[1]}, "
              f"w={best_params[2]}, mu={best_params[3]}, k1={best_params[4]}, b={best_params[5]} -> MAP={best_map:.6f}")