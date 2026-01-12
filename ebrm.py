# entity_extraction.py
from __future__ import annotations

import re
import json
import html
from pathlib import Path
from typing import Optional, Dict, Any, List

import spacy
from pyserini.search.lucene import LuceneSearcher


_TAG_RE = re.compile(r"<[^>]+>")
_POSSESSIVE_RE = re.compile(r"('s|’s)$", re.IGNORECASE)
_LEADING_DET_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_MULTI_WS_RE = re.compile(r"\s+")

DEFAULT_KEEP_LABELS = {
    "PERSON", "ORG", "GPE", "LOC",
    "NORP", "FAC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE"
}

# DEFAULT_KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC"}



def trec_raw_to_text(raw: str) -> str:
    """Best-effort stripping of TREC Robust04-style markup to plain text."""
    if not raw:
        return ""
    raw = html.unescape(raw)
    text = _TAG_RE.sub(" ", raw)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def normalize_entity(text: str, drop_leading_det: bool = True) -> str:
    """
    Light normalization to reduce NER noise:
    - strip trailing possessive ('s / ’s)
    - optionally remove leading determiners (the/a/an)
    - collapse whitespace
    """
    t = text.strip()
    t = _POSSESSIVE_RE.sub("", t).strip()
    if drop_leading_det:
        t = _LEADING_DET_RE.sub("", t).strip()
    t = _MULTI_WS_RE.sub(" ", t).strip()
    return t


def extract_entities_to_jsonl(
    index_name: str = "robust04",
    out_path: str = "robust04_entities.jsonl",
    spacy_model: str = "en_core_web_sm",
    keep_labels: Optional[set[str]] = None,
    batch_size: int = 128,
    n_process: int = 1,
    limit_docs: Optional[int] = None,
    progress_every: int = 5000,
    drop_leading_det: bool = True,
    min_entity_len: int = 3,
) -> Dict[str, Any]:
    """
    Iterates documents in a Lucene index via LuceneSearcher (pyserini 0.36 compatible),
    extracts NER entities with spaCy, and writes JSONL:
      {"docid": "...", "entities": [[text, label], ...]}

    Notes:
    - Uses manual batching (no as_tuples) to avoid spaCy version differences.
    - 'docid' is the external collection docid (e.g., FBIS3-10555, LA..., FR...).
    """
    keep_labels = keep_labels or DEFAULT_KEEP_LABELS

    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    num_docs = int(searcher.num_docs)  # internal Lucene ids: 0..num_docs-1
    n = num_docs if limit_docs is None else min(num_docs, int(limit_docs))

    # Load spaCy model; disable unused components to speed up.
    nlp = spacy.load(
        spacy_model,
        disable=["tagger", "parser", "lemmatizer", "attribute_ruler"]
    )
    if "ner" not in nlp.pipe_names:
        raise RuntimeError(f"Model has no NER pipe. pipe_names={nlp.pipe_names}")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    total_entities = 0

    def tuples_stream():
        for internal_id in range(n):
            doc = searcher.doc(internal_id)
            if doc is None:
                continue
            ext_docid = doc.docid()
            raw = doc.raw()
            text = trec_raw_to_text(raw)
            if text:
                yield ext_docid, text

    with out_file.open("w", encoding="utf-8") as f:
        batch_docids: List[str] = []
        batch_texts: List[str] = []

        def flush_batch():
            nonlocal written, total_entities, batch_docids, batch_texts
            if not batch_texts:
                return

            docs = nlp.pipe(batch_texts, batch_size=batch_size, n_process=n_process)

            for docid, doc in zip(batch_docids, docs):
                ents: List[List[str]] = []
                seen = set()

                for ent in doc.ents:
                    if ent.label_ not in keep_labels:
                        continue

                    norm = normalize_entity(ent.text, drop_leading_det=drop_leading_det)
                    if not norm:
                        continue

                    # Drop very long entities (often noisy headlines)
                    if len(norm) > 60:
                        continue

                    # Drop short ALL-CAPS tokens (acronyms can be OK; tune as you like)
                    if norm.isupper() and len(norm) <= 5:
                        continue

                    # Drop tokens that are just punctuation / digits
                    if not any(c.isalpha() for c in norm):
                        continue

                    if min_entity_len and len(norm) < min_entity_len:
                        continue

                    key = (norm.lower(), ent.label_)
                    if key in seen:
                        continue
                    seen.add(key)

                    ents.append([norm, ent.label_])

                total_entities += len(ents)
                f.write(json.dumps({"docid": docid, "entities": ents}, ensure_ascii=False) + "\n")
                written += 1

                if progress_every and written % progress_every == 0:
                    print(f"[entity_extraction] wrote {written} docs...")

            batch_docids = []
            batch_texts = []

        for docid, text in tuples_stream():
            batch_docids.append(docid)
            batch_texts.append(text)
            if len(batch_texts) >= batch_size:
                flush_batch()

        flush_batch()

    return {
        "index_name": index_name,
        "out_path": str(out_file.resolve()),
        "docs_written": written,
        "total_entities": total_entities,
        "avg_entities_per_doc": (total_entities / written) if written else 0.0,
        "num_docs_in_index": num_docs,
        "num_docs_processed": n,
        "batch_size": batch_size,
        "n_process": n_process,
        "drop_leading_det": drop_leading_det,
        "min_entity_len": min_entity_len,
        "spacy_model": spacy_model,
    }
