import os
from dotenv import load_dotenv
load_dotenv()
os.environ["JAVA_HOME"] = os.getenv("JAVA_HOME")
from tqdm import tqdm
from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
from processing import split_passages, clean_robust
from sentence_transformers import CrossEncoder, SentenceTransformer
import re
from collections import defaultdict
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CROSS_ENCODER = CrossEncoder(os.getenv("CROSS_ENCODER"), device=DEVICE)

class SearchEngine:
    def __init__(self):
        self.reader = IndexReader.from_prebuilt_index('robust04')
        self.searcher = LuceneSearcher.from_prebuilt_index('robust04')
        # Other approaches here

    def set_searcher(self, approach="qld", fb_terms=5, fb_docs=10, original_query_weight=0.8, mu=1000):
        if approach=="qld":
            # Setting query likelihood with dirichlet prior
            self.searcher.set_qld(mu=mu)
            # Setting RM3 expanding the query, with a safe alpha
            self.searcher.set_rm3(fb_terms=fb_terms, fb_docs=fb_docs, original_query_weight=original_query_weight)
        elif approach=="bm25":
            self.searcher.set_bm25(k1=0.9, b=0.4)

    def get_top_k(self, query, k=5, clean=True  ):
        """
        Get the top k ranked (full) documents using the searcher
        :param query: the query
        :param k: top results to retrieve (default: 5)
        :param clean: Whether to clean the retrieved docs and extract metadata (default: True)
        :return:
        """
        context = []
        hits = self.searcher.search(query, k)
        # Get text from hits
        for hit in hits:
            doc = self.searcher.doc(hit.docid)
            raw_doc = doc.raw()
            if clean:
                cleaned_doc, doc_metadata = clean_robust(raw_doc)
                doc_metadata['docid'] = hit.docid
                doc_metadata['score'] = hit.score
                context.append((cleaned_doc, doc_metadata))
            else:
                context.append((raw_doc, {'docid': hit.docid, 'score':hit.score}))
        return context

    def rerank(self, query, retrieval_candidates, max_weight=0.8, reranker=CROSS_ENCODER):
        def _remove_whitespaces(text):
            WS_NEWLINES = re.compile(r"\s*\n\s*")
            WS_SPACES = re.compile(r"[ \t]+")
            text = WS_NEWLINES.sub(" ", text)
            text = WS_SPACES.sub(" ", text)
            return text
        def _collated_doc_score(scores, max_weight=0.8):
            return max(scores)*max_weight + (1-max_weight)*(sum(scores)-max(scores))/(len(scores)-1)
        pairs = [[query, _remove_whitespaces(doc.page_content)] for doc in retrieval_candidates]
        cross_scores = reranker.predict(pairs)
        per_doc_scores = defaultdict(list)
        for score, doc in zip(cross_scores, retrieval_candidates):
            per_doc_scores[doc.metadata['docid']].append(score)
        collated_doc_scores = {}
        for docid, scores in per_doc_scores.items():
            collated_doc_scores[docid] = _collated_doc_score(scores, max_weight=max_weight)
        ranked = sorted(collated_doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def retrieve_rerank(self, query, k=1000, m=100):
        assert k>=m, "initial retrieval k must be bigger-equal than fine reranker m"
        hits = self.get_top_k(query, k, clean=True) # Hits are score-sorted by default
        top_m = hits[:m]
        passages_top_m = split_passages(top_m)
        top_docs_reranked = self.rerank(query, passages_top_m)
        all_docs = top_docs_reranked + [(hit[1]["docid"], hit[1]["score"]) for hit in hits[m:]]
        return all_docs


    def search_and_write_trec_run(self, query, k, topic_id, run_tag, output_file, m=100):
        hits = self.retrieve_rerank(query, k, m)
        with open(output_file, "a", encoding="utf-8") as f:
            for rank, hit in enumerate(hits, start=1):
                f.write(
                    f"{topic_id} Q0 {hit.docid} {rank} {hit.score:.6f} {run_tag}\n"
                )

    def search_all_queries(self, topics, k=1000, run_tag="run1", output_file="run.txt", m=100):
        """
        Search all queries according to topics list
        :param topics: list of [(query id, query]
        :param k: top results to retrieve (default: 1000)
        :param run_tag: name of run to write as the format
        :param output_file: name of outputfile (default: run.txt)
        :param m: reranking threshold (default: 100)
        :return:
        """

        # Clear the file
        with open(output_file, "w", encoding="utf-8") as f:
            pass

        for qid, query in topics.items():
            self.search_and_write_trec_run(query, k, qid, run_tag, output_file, m=m)
