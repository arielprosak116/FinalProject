# entity_extraction_subset.py
from __future__ import annotations

import json
import re
import html
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict, Any

import spacy
from pyserini.search.lucene import LuceneSearcher


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_POSSESSIVE_RE = re.compile(r"('s|’s|')$", re.IGNORECASE)
_LEADING_DET_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_HOME_EDITION_RE = re.compile(r"^home edition\b", re.IGNORECASE)

DEFAULT_KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC"}  # best default for EBRM


def trec_raw_to_text(raw: str) -> str:
    if not raw:
        return ""
    raw = html.unescape(raw)
    text = _TAG_RE.sub(" ", raw)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_entity(text: str, drop_leading_det: bool = True) -> str:
    t = text.strip()
    t = _POSSESSIVE_RE.sub("", t).strip()
    if drop_leading_det:
        t = _LEADING_DET_RE.sub("", t).strip()
    t = _WS_RE.sub(" ", t).strip()
    return t


def load_docids(docids_path: str, limit: Optional[int] = None) -> List[str]:
    docids = []
    with open(docids_path, "r", encoding="utf-8") as f:
        for line in f:
            d = line.strip()
            if d:
                docids.append(d)
            if limit is not None and len(docids) >= limit:
                break
    return docids


def extract_entities_for_docids_to_jsonl(
    docids_path: str,
    out_path: str = "robust04_entities_subset.jsonl",
    index_name: str = "robust04",
    spacy_model: str = "en_core_web_sm",
    keep_labels: Optional[Set[str]] = None,
    batch_size: int = 256,
    n_process: int = 1,
    progress_every: int = 2000,
    drop_leading_det: bool = True,
    min_entity_len: int = 3,
    max_entity_len: int = 60,
) -> Dict[str, Any]:
    """
    Writes JSONL: {"docid": "...", "entities": [[text,label], ...]}
    """
    keep_labels = keep_labels or DEFAULT_KEEP_LABELS
    docids = load_docids(docids_path)

    searcher = LuceneSearcher.from_prebuilt_index(index_name)

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

    batch_ids: List[str] = []
    batch_texts: List[str] = []

    def flush_batch(fh):
        nonlocal written, total_entities, batch_ids, batch_texts
        if not batch_texts:
            return
        docs = nlp.pipe(batch_texts, batch_size=batch_size, n_process=n_process)
        for docid, doc in zip(batch_ids, docs):
            ents: List[List[str]] = []
            seen = set()
            for ent in doc.ents:
                if ent.label_ not in keep_labels:
                    continue
                norm = normalize_entity(ent.text, drop_leading_det=drop_leading_det)
                if not norm:
                    continue
                if _HOME_EDITION_RE.match(norm):
                    continue
                if min_entity_len and len(norm) < min_entity_len:
                    continue
                if max_entity_len and len(norm) > max_entity_len:
                    continue
                if not any(c.isalpha() for c in norm):
                    continue
                key = (norm.lower(), ent.label_)
                if key in seen:
                    continue
                seen.add(key)
                ents.append([norm, ent.label_])

            total_entities += len(ents)
            fh.write(json.dumps({"docid": docid, "entities": ents}, ensure_ascii=False) + "\n")
            written += 1

            if progress_every and written % progress_every == 0:
                print(f"[entity_subset] wrote {written} docs...")

        batch_ids = []
        batch_texts = []

    with out_file.open("w", encoding="utf-8") as f:
        for docid in docids:
            doc = searcher.doc(docid)
            if doc is None:
                continue
            text = trec_raw_to_text(doc.raw())
            if not text:
                continue

            batch_ids.append(docid)
            batch_texts.append(text)
            if len(batch_texts) >= batch_size:
                flush_batch(f)

        flush_batch(f)

    return {
        "out_path": str(out_file.resolve()),
        "docids_input": len(docids),
        "docs_written": written,
        "total_entities": total_entities,
        "avg_entities_per_doc": (total_entities / written) if written else 0.0,
        "batch_size": batch_size,
        "n_process": n_process,
        "labels": sorted(list(keep_labels)),
    }
