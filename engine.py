# search_engine.py
import os
import math
import pickle
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
os.environ["JAVA_HOME"] = os.getenv("JAVA_HOME")

from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher

from entity_store import EntityStore


class SearchEngine:
    def __init__(self, entities_jsonl=None, entities_offsets=None):
        self.reader = IndexReader.from_prebuilt_index('robust04')
        self.searcher = LuceneSearcher.from_prebuilt_index('robust04')

        # Optional: EBRM entity store (docid -> entities)
        self.entity_store = None
        if entities_jsonl and entities_offsets:
            self.entity_store = EntityStore.load(entities_jsonl, entities_offsets)

    def set_searcher(self, approach="qld", fb_terms=5, fb_docs=10, original_query_weight=0.8, mu=1000, k1=0.9, b=0.4):
        if approach == "qld":
            self.searcher.set_qld(mu=mu)
            self.searcher.set_rm3(
                fb_terms=fb_terms,
                fb_docs=fb_docs,
                original_query_weight=original_query_weight
            )
        elif approach == "bm25":
            self.searcher.set_bm25(k1=k1, b=b)
            self.searcher.set_rm3(
                fb_terms=fb_terms,
                fb_docs=fb_docs,
                original_query_weight=original_query_weight)

    def get_top_k(self, query, k=5, to_chunk=False):
        context = []
        hits = self.searcher.search(query, k)
        for hit in hits:
            doc = self.searcher.doc(hit.docid)
            raw_doc = doc.raw()
            context.append((hit.docid, raw_doc, hit.score))
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
                f.write(f"{topic_id} Q0 {hit.docid} {rank} {hit.score:.6f} {run_tag}\n")

    def search_all_queries(self, topics, k=1000, run_tag="run1", output_file="run.txt"):
        with open(output_file, "w", encoding="utf-8") as f:
            pass
        for qid, query in topics.items():
            self.search_and_write_trec_run(query, k, qid, run_tag, output_file)

    # ---------------- EBRM INTEGRATION ----------------

    @staticmethod
    def _softmax(scores, temp=1.0):
        if not scores:
            return []
        mx = max(scores)
        exps = [math.exp((s - mx) / max(1e-9, temp)) for s in scores]
        Z = sum(exps)
        return [e / Z for e in exps]

    @staticmethod
    def _lucene_escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def _ebrm_rank_entities(
            self,
            hits,
            fb_docs=5,
            fb_entities=20,
            allowed_labels=None,
            temperature=1.0,
    ):
        """
        Build an entity relevance model (EBRM) from the top fb_docs hits.

        Steps:
          1) Take top fb_docs hits
          2) Convert their baseline scores into P(d|Q) via softmax
          3) Accumulate entity weights: score(e) += P(d|Q) for each entity appearing in doc d
          4) (Optional) Apply subset-IDF penalty using self.entity_store.idf(entity_lower)
          5) Return top fb_entities as [(entity_text, weight), ...]

        Notes:
          - Requires self.entity_store to be loaded.
          - Entities are de-duplicated per document (an entity counted once per doc).
          - allowed_labels can restrict entity types (e.g., {"ORG","PERSON"}).
          - temperature controls softmax sharpness (higher -> flatter).
        """

        if self.entity_store is None:
            raise RuntimeError("EBRM requested but entity_store is not loaded.")

        if not hits:
            return []

        import math
        from collections import defaultdict

        top = hits[:fb_docs]

        # --- 1) Softmax over top-doc scores to approximate P(d|Q) ---
        scores = [float(h.score) for h in top]
        if temperature <= 0:
            temperature = 1.0

        # Stabilized softmax: exp((s - max)/temp)
        mx = max(scores)
        exps = [math.exp((s - mx) / temperature) for s in scores]
        Z = sum(exps) + 1e-12
        p_docs = [e / Z for e in exps]

        # --- 2) Accumulate entity weights ---
        ent_scores = defaultdict(float)  # entity_lower -> score
        ent_casing = {}  # entity_lower -> first-seen original casing (for nicer output)

        for h, p_d in zip(top, p_docs):
            ents = self.entity_store.get(h.docid)  # list[(entity_text,label)] or []
            seen = set()

            for ent_text, ent_label in ents:
                if allowed_labels and ent_label not in allowed_labels:
                    continue

                key = str(ent_text).strip().lower()
                if not key:
                    continue
                if key in seen:
                    continue
                seen.add(key)

                ent_scores[key] += float(p_d)
                if key not in ent_casing:
                    ent_casing[key] = str(ent_text).strip()

        if not ent_scores:
            return []

        # --- 3) Apply cheap "IDF-like" penalty based on subset DF (optional but recommended) ---
        # idf(e) = log((N+1)/(df(e)+1)) computed over your extracted subset JSONL
        if getattr(self.entity_store, "entity_df", None) is not None and getattr(self.entity_store, "N", 0) > 0:
            for e in list(ent_scores.keys()):
                ent_scores[e] *= float(self.entity_store.idf(e))

        # --- 4) Select top fb_entities ---
        ranked = sorted(ent_scores.items(), key=lambda kv: kv[1], reverse=True)[:fb_entities]

        # return in nice casing
        return [(ent_casing.get(e, e), float(w)) for e, w in ranked]

    def _build_query_with_entities(self, original_query: str, entity_expansions, entity_weight: float = 0.25):
        """
        Add boosted entity phrases to Lucene query string.
        """
        # print("OG query: ")
        # print(original_query)
        parts = [original_query.strip()]
        for ent_text, w in entity_expansions:
            boost = entity_weight * float(w)
            if boost <= 0:
                continue
            t = self._lucene_escape(ent_text.strip())
            if not t:
                continue
            if " " in t:
                parts.append(f"\"{t}\"^{boost:.6f}")
            else:
                parts.append(f"{t}^{boost:.6f}")
        new_query = " ".join(parts).strip()
        # print(new_query)
        return new_query

    def search_and_write_trec_run_hybrid_ebrm(
        self,
        query,
        k,
        topic_id,
        run_tag,
        output_file,
        fb_docs=5,
        fb_entities=10,
        entity_weight=0.25,
        allowed_labels=None,
    ):
        """
        Two-stage:
          1) retrieve with current searcher config (can include RM3)
          2) compute EBRM entities from top fb_docs
          3) rerun with query + entity boosts
        """
        hits = self.searcher.search(query, k)

        ent_exp = self._ebrm_rank_entities(
            hits=hits,
            fb_docs=fb_docs,
            fb_entities=fb_entities,
            allowed_labels=allowed_labels,
        )

        hybrid_query = self._build_query_with_entities(query, ent_exp, entity_weight=entity_weight)

        hits2 = self.searcher.search(hybrid_query, k)
        hits_sorted = sorted(hits2, key=lambda h: h.score, reverse=True)

        with open(output_file, "a", encoding="utf-8") as f:
            for rank, hit in enumerate(hits_sorted, start=1):
                f.write(f"{topic_id} Q0 {hit.docid} {rank} {hit.score:.6f} {run_tag}\n")

    def search_all_queries_hybrid_ebrm(
        self,
        topics,
        k=1000,
        run_tag="run_hybrid_ebrm",
        output_file="run_hybrid_ebrm.txt",
        fb_docs=5,
        fb_entities=10,
        entity_weight=0.25,
        allowed_labels=None,
    ):
        with open(output_file, "w", encoding="utf-8") as f:
            pass
        for qid, query in topics.items():
            self.search_and_write_trec_run_hybrid_ebrm(
                query=query,
                k=k,
                topic_id=qid,
                run_tag=run_tag,
                output_file=output_file,
                fb_docs=fb_docs,
                fb_entities=fb_entities,
                entity_weight=entity_weight,
                allowed_labels=allowed_labels,
            )

    def _ebrm_rerank_hits(
            self,
            hits,
            fb_docs=5,
            fb_entities=20,
            alpha=0.9,
            allowed_labels=None,
    ):
        """
        Rerank candidate docs using a calibrated mixture of:
          - baseline scores (BM25+RM3 / QLD+RM3 etc.)
          - entity overlap scores from an EBRM entity relevance model

        final(d) = alpha * zscore(baseline(d)) + (1-alpha) * zscore(entity_overlap(d))

        entity_overlap(d) = sum_{e in ents(d) ∩ top_entities} weight(e)

        Notes:
        - Requires self.entity_store to be loaded (docid -> list[(entity_text, label)])
        - Uses only entities from top fb_docs docs to build the entity model.
        """

        if self.entity_store is None:
            raise RuntimeError("EBRM requested but entity_store is not loaded.")

        if not hits:
            return []

        # --- helpers ---
        import math

        def zscore(xs):
            if not xs:
                return []
            m = sum(xs) / len(xs)
            v = sum((x - m) ** 2 for x in xs) / len(xs)
            s = math.sqrt(v) + 1e-9
            return [(x - m) / s for x in xs]

        # --- 1) Build entity relevance model from top fb_docs docs ---
        ent_exp = self._ebrm_rank_entities(
            hits=hits,
            fb_docs=fb_docs,
            fb_entities=fb_entities,
            allowed_labels=allowed_labels,
        )
        # dict: entity_lower -> weight
        ent_w = {e.lower(): float(w) for e, w in ent_exp}

        # --- 2) Compute per-doc entity overlap scores for all candidate hits ---
        e_scores = []
        docids = []
        base_scores = []

        for h in hits:
            docids.append(h.docid)
            base_scores.append(float(h.score))

            ents = self.entity_store.get(h.docid)  # list[(text,label)] or []
            seen = set()
            score = 0.0

            for ent_text, ent_label in ents:
                if allowed_labels and ent_label not in allowed_labels:
                    continue
                key = ent_text.lower()
                if key in seen:
                    continue
                seen.add(key)
                score += ent_w.get(key, 0.0)

            e_scores.append(score)

        # --- 3) Calibrate both components to comparable scales (z-score per query) ---
        base_z = zscore(base_scores)
        ent_z = zscore(e_scores)

        # --- 4) Mix and sort ---
        reranked = []
        for docid, b, e in zip(docids, base_z, ent_z):
            final = alpha * b + (1.0 - alpha) * e
            reranked.append((docid, final))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def search_and_write_trec_run_ebrm_rerank(
            self,
            query, k, topic_id, run_tag, output_file,
            fb_docs=5, fb_entities=20, alpha=0.8, allowed_labels=None
    ):
        # IMPORTANT: use your strong baseline retrieval here (QLD+RM3 works great)
        hits = self.searcher.search(query, k)

        reranked = self._ebrm_rerank_hits(
            hits=hits,
            fb_docs=fb_docs,
            fb_entities=fb_entities,
            alpha=alpha,
            allowed_labels=allowed_labels
        )

        with open(output_file, "a", encoding="utf-8") as f:
            for rank, (docid, score) in enumerate(reranked, start=1):
                f.write(f"{topic_id} Q0 {docid} {rank} {score:.6f} {run_tag}\n")

    def search_all_queries_ebrm_rerank(
            self,
            topics, k=1000,
            run_tag="rm3_then_ebrm_rerank",
            output_file="run_ebrm_rerank.txt",
            fb_docs=5, fb_entities=20, alpha=0.8,
            allowed_labels=None
    ):
        with open(output_file, "w", encoding="utf-8") as f:
            pass
        for qid, query in topics.items():
            self.search_and_write_trec_run_ebrm_rerank(
                query=query, k=k, topic_id=qid,
                run_tag=run_tag, output_file=output_file,
                fb_docs=fb_docs, fb_entities=fb_entities, alpha=alpha,
                allowed_labels=allowed_labels
            )

