import os
from dotenv import load_dotenv
load_dotenv()
os.environ["JAVA_HOME"] = os.getenv("JAVA_HOME")
from tqdm import tqdm
from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
from processing import split_passages, clean_robust, Hit
from sentence_transformers import CrossEncoder
from mxbai_rerank import MxbaiRerankV2
from inranker import T5Ranker
import re
from collections import defaultdict
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
CROSS_ENCODER = os.getenv("CROSS_ENCODER")
SUPPORTED_RERANKERS = ["CE", "mxbai", "monot5", "twolar", "inranker"]



def weighted_rrf_fuse(runs, weights=None, rrf_k=60, save_text=False):
    """
    runs: list[list[Hit]] docids ordered best->worst
    weights: list[float] same length as runs, defaults to 1/len(runs) each
    save_text: Save the text field (relevant if this isn't the final step)
    """
    assert sum(weights) == 1.0, "Weights must sum to 1.0"
    if weights is None:
        weights = [1/len(runs)] * len(runs)
    scores = defaultdict(float)
    texts = defaultdict(str)
    for run, w in zip(runs, weights):
        for rank, hit in enumerate(run, start=1):
            scores[hit.docid] += w * (1.0 / (rrf_k + rank))
            texts[hit.docid] = hit.text if save_text else None

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [Hit(docid=docid, score=score, text=texts[docid]) for docid, score in fused]



class Reranker:
    def __init__(self, reranker_type, device=DEVICE):
        self.reranker_type = reranker_type
        if self.reranker_type not in SUPPORTED_RERANKERS:
            raise ValueError(f"reranker_type must be in {SUPPORTED_RERANKERS}")
        if self.reranker_type == 'CE':
            self.model = CrossEncoder(CROSS_ENCODER, device=DEVICE)
        elif self.reranker_type == 'mxbai':
            self.model = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-large-v2", device=device)
            print(device)
            self.model.to(device)
        elif self.model = T5Ranker(model_name_or_path="unicamp-dl/InRanker-3B")

        else:
            raise NotImplementedError("Type not implemented yet sry :(")

    def rerank(self, query, retrieval_candidates, max_weight=0.8):
        def _remove_whitespaces(text):
            WS_NEWLINES = re.compile(r"\s*\n\s*")
            WS_SPACES = re.compile(r"[ \t]+")
            text = WS_NEWLINES.sub(" ", text)
            text = WS_SPACES.sub(" ", text)
            return text

        def _collated_doc_score(scores, max_weight=0.8):
            return max(scores) * max_weight + (1 - max_weight) * (sum(scores) - max(scores)) / (len(scores) - 1) if len(scores) > 1 else max(scores)

        cleaned_docs = [_remove_whitespaces(doc.page_content) for doc in retrieval_candidates]
        per_doc_scores = defaultdict(list)
        if self.reranker_type == 'CE':
            pairs = [[query, cleaned_doc] for cleaned_doc in cleaned_docs]
            cross_scores = self.model.predict(pairs)
            for score, doc in zip(cross_scores, retrieval_candidates):
                per_doc_scores[doc.metadata['docid']].append(score)
        if self.reranker_type == 'mxbai':
            id2doc = {i:doc.metadata['docid'] for i, doc in enumerate(retrieval_candidates)}
            cross_scores = self.model.rank(query, cleaned_docs, return_documents=False)
            for score in cross_scores:
                per_doc_scores[id2doc[score.index]].append(score.score)

        collated_doc_scores = {}
        for docid, scores in per_doc_scores.items():
            collated_doc_scores[docid] = _collated_doc_score(scores, max_weight=max_weight)
        ranked = sorted(collated_doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [Hit(docid=doc[0], score=doc[1]) for doc in ranked]



class SearchEngine:
    def __init__(self):
        self.reader = IndexReader.from_prebuilt_index('robust04')
        self.searcher = LuceneSearcher.from_prebuilt_index('robust04')
        self.reranker = None

    def set_searcher(self, approach="qld", fb_terms=5, fb_docs=10, original_query_weight=0.8, mu=1000, reranker='CE'):
        if approach=="qld":
            # Setting query likelihood with dirichlet prior
            self.searcher.set_qld(mu=mu)
            # Setting RM3 expanding the query, with a safe alpha
            self.searcher.set_rm3(fb_terms=fb_terms, fb_docs=fb_docs, original_query_weight=original_query_weight)
        elif approach=="bm25":
            self.searcher.set_bm25(k1=0.5, b=0.36)
            self.searcher.set_rm3(fb_terms=fb_terms,fb_docs=fb_docs,original_query_weight=original_query_weight)
        if reranker is not None:
            self.reranker = Reranker(reranker)

    def get_top_k(self, query, k=5, clean=True, qid=None):
        """
        Get the top k ranked (full) documents using the searcher
        :param query: the query
        :param k: top results to retrieve (default: 5)
        :param clean: Whether to clean the retrieved docs and extract metadata (default: True)
        :param qid: query id
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
                context.append(Hit(qid=qid, query=query, docid=hit.docid, score=hit.score, meta=doc_metadata, text=cleaned_doc))
            else:
                context.append(Hit(qid=qid, query=query, docid=hit.docid, score=hit.score, text=raw_doc))
        return context


    def multi_query_fuse(self, qid, topics_list, llm_query_fusion_weights, k=1000):
        top_ks = []
        assert len(llm_query_fusion_weights) == len(topics_list), "Weight & lists mismatch"
        if len(topics_list) > 1:
            for i, topics in enumerate(topics_list):
                if llm_query_fusion_weights[i] == 0:
                    continue
                query = topics[qid]
                top_ks.append(self.get_top_k(query, k, clean=True, qid=qid))
            top_k_fused = weighted_rrf_fuse(top_ks, weights=llm_query_fusion_weights, save_text=True)
            return top_k_fused
        else:
            return self.get_top_k(topics_list[0][qid], k, clean=True, qid=qid)

    def retrieve_rerank(self, query, hits, m=100, fusion_weights=None):
        top_m = hits[:m]
        passages_top_m = split_passages(top_m)
        if self.reranker:
            top_m_reranked = self.reranker.rerank(query, passages_top_m)
            top_m_fused_permutations = [weighted_rrf_fuse([top_m_reranked, top_m], weights=[1-fusion_weight,fusion_weight]) for fusion_weight in fusion_weights]
            all_docs_reranked = [top_m_fused + hits[m:] for top_m_fused in top_m_fused_permutations]
            return all_docs_reranked
        else:
            return [hits]


    def search_and_write_trec_run(self, query, k, topic_id, run_tag, output_file, fusion_weights=None,
                                  query_weights=None,
                                  topics_lists=None,
                                  m=100):
        if fusion_weights is None:
            fusion_weights = [0]
        assert k >= m, "initial retrieval k must be bigger-equal than fine reranker m"
        hits = self.multi_query_fuse(topic_id, topics_lists, query_weights, k=k)  # Hits are score-sorted by default
        hits_per_fusion_weight = self.retrieve_rerank(query, hits, m, fusion_weights)
        for i, hits in enumerate(hits_per_fusion_weight):
            with open(f"Results/{output_file}_rrf_{fusion_weights[i]}.txt", "a", encoding="utf-8") as f:
                for rank, hit in enumerate(hits, start=1):
                    f.write(
                        f"{topic_id} Q0 {hit.docid} {rank} {hit.score:.6f} {run_tag}\n"
                    )


    def search_all_queries(self, topics_lists, k=1000, run_tag="run1", output_file="run.txt", m=100,
                           llm_query_fusion_weights=None,
                           rerank_fusion_weights=None):
        """
        Search all queries according to topics list
        :param topics_lists: list of [(query id, query] for topic in topics. Each topic is taken form a .txt listing all queries.
        :param k: top results to retrieve (default: 1000)
        :param run_tag: name of run to write as the format
        :param output_file: name of outputfile (default: run.txt)
        :param m: reranking threshold (default: 100)
        :param llm_query_fusion_weights: list of fusion weights on multiple query ablations (default: [1,0...,0])
        :param rerank_fusion_weights: rrf weights to experiment with (default: 0)
        """
        if rerank_fusion_weights is None:
            rerank_fusion_weights = [0]
        if llm_query_fusion_weights is None:
            llm_query_fusion_weights = [1]+[0]*(len(topics_lists) - 1)

        # TODO assuming topics_list is [topics] this should work. Make multiple llm fuse work
        # TODO also try reranking 400 not 100 see if shit changes.
        for qid, query in tqdm(topics_lists[0].items(), desc="Searching topics"):
            self.search_and_write_trec_run(query, k, qid, run_tag, output_file, m=m,
                                           fusion_weights=rerank_fusion_weights, query_weights=llm_query_fusion_weights,
                                           topics_lists=topics_lists)
