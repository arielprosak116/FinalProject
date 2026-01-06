import os
import json
from dotenv import load_dotenv
load_dotenv()
os.environ["JAVA_HOME"] = os.getenv("JAVA_HOME")
from tqdm import tqdm
from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
import pickle

class SearchEngine:
    def __init__(self):
        self.reader = IndexReader.from_prebuilt_index('robust04')
        self.searcher = LuceneSearcher.from_prebuilt_index('robust04')
        # Other approaches here

    def set_searcher(self, approach="qld", fb_terms=5, fb_docs=10, original_query_weight=0.8):
        if approach=="qld":
            # Setting query likelihood with dirichle of 1000
            self.searcher.set_qld(mu=1000)
            # Setting RM3 expanding the query, with a safe alpha
            self.searcher.set_rm3(fb_terms=fb_terms, fb_docs=fb_docs, original_query_weight=original_query_weight)
        elif approach=="bm25":
            self.searcher.set_bm25(k1=0.9, b=0.4)

    def get_top_k(self, query, k=5, to_chunk=False):
        """
        Get the top k ranked (full) documents using the searcher
        :param query: the query
        :param k: top results to retrieve (default: 5)
        :param to_chunk: To chunk or not to chunk (default: False for now)
        :return:
        """
        context = []
        hits = self.searcher.search(query, k)
        # Get text from hits
        for hit in hits:
            doc = self.searcher.doc(hit.docid)
            raw_doc = doc.raw()
            # TODO proper parsing
            context.append((hit.docid, raw_doc, hit.score))
        if not to_chunk:
            return context
        else: # Passage retrieval stuff
            # all_paragraphs = clean_split_wiki_docs(context, context_metadatas)
            #return all_paragraphs
            return context

    def get_context_batch(self, samples, k=5, save_name=None):
        all_contexts = {}
        for query in tqdm(samples['question']):
            all_contexts[query] = self.get_top_k(query, k, to_chunk=False)
        if save_name:
            with open(f'top{k}_full_docs{save_name}.pkl', 'wb') as f:
                pickle.dump(all_contexts, f)
        else:
            print("Did not provide save_name, no pkl will be created")
        return all_contexts

    def search_and_write_trec_run(self, query, k, topic_id, run_tag, output_file):
        hits = self.searcher.search(query, k)
        hits_sorted = sorted(hits, key=lambda h: h.score, reverse=True)
        with open(output_file, "a", encoding="utf-8") as f:
            for rank, hit in enumerate(hits_sorted, start=1):
                f.write(
                    f"{topic_id} Q0 {hit.docid} {rank} {hit.score:.6f} {run_tag}\n"
                )

    def search_all_queries(self, topics, k=1000, run_tag="run1", output_file="run.txt"):
        """
        Search all queries according to topics list
        :param topics: list of [(query id, query]
        :param k: top results to retrieve (default: 1000)
        :param run_tag: name of run to write as the format
        :param output_file: name of outputfile (default: run.txt)
        :return:
        """

        # Clear the file
        with open(output_file, "w", encoding="utf-8") as f:
            pass

        for qid, query in topics.items():
            self.search_and_write_trec_run(query, k, qid, run_tag, output_file)
