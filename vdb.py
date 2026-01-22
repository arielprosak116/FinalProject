import faiss
import numpy as np
from typing import List, Optional, Tuple, Iterable
from pathlib import Path
import json
from dataclasses import dataclass
from collections import defaultdict
from engine import SearchEngine
from processing import Hit, split_passages, clean_robust
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class Passage:
    docid: str
    text: str
    passage_idx: int


@dataclass
class BuildState:
    next_docnum: int = 0          # resume pointer in Lucene docnum space
    total_passages: int = 0       # how many passage vectors were added

    def to_dict(self) -> dict:
        return {"next_docnum": self.next_docnum, "total_passages": self.total_passages}

    @classmethod
    def from_dict(cls, d: dict) -> "BuildState":
        return cls(
            next_docnum=int(d.get("next_docnum", 0)),
            total_passages=int(d.get("total_passages", 0)),)


def _doc_text_from_raw(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.startswith("{") and raw.endswith("}"):
        try:
            j = json.loads(raw)
            for key in ("contents", "text", "body", "abstract", "title"):
                if key in j and isinstance(j[key], str) and j[key].strip():
                    return j[key].strip()
        except Exception:
            pass
    return raw


def _load_state(state_path: Path) -> BuildState:
    if not state_path.exists():
        return BuildState()
    return BuildState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))


def _save_state(state_path: Path, state: BuildState) -> None:
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(state_path)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms

def hits_to_passages(
    hits: list[Hit],
) -> list[Passage]:
    """
    Uses the existing split_passages(hits) function.
    """
    docs = split_passages(hits)

    passages: list[Passage] = []
    for d in docs:
        docid = d.metadata.get("docid") or d.metadata.get("id")
        passages.append(
            Passage(
                docid=str(docid),
                text=d.page_content,
                passage_idx=d.metadata.get("passage_idx", 0),
            )
        )
    return passages

class VdbFaiss:
    """
    FAISS vdb naive implementation
    """
    def __init__(
        self,
        embedding_function,
        *,
        index_factory: str = "HNSW32",
        metric: str = "ip",  # "ip" (cosine/dot) or "l2"
        normalize_for_cosine: bool = True,
    ):
        self.embedding_function = embedding_function
        self.dim = int(np.asarray(self.embedding_function.encode(["test"])).shape[1])

        self.metric = metric.lower()
        self.normalize_for_cosine = normalize_for_cosine and (self.metric == "ip")

        faiss_metric = faiss.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss.METRIC_L2
        self.index = faiss.index_factory(self.dim, index_factory, faiss_metric)

        # Persisted metadata: one line per vector added
        # {"vector_id": <int>, "docid": "...", "passage_idx": <int>}
        self._meta_path: Optional[Path] = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        embs = np.asarray(self.embedding_function.encode(texts), dtype=np.float32)
        if embs.ndim == 1:
            embs = embs[None, :]
        if self.normalize_for_cosine:
            embs = _normalize_rows(embs)
        return embs

    def add_passages(self, passages: List[Passage]) -> Tuple[int, int]:
        """
        Adds passages to FAISS and appends metadata lines.
        Returns (start_vector_id, num_added).
        """
        if not passages:
            return int(self.index.ntotal), 0

        texts = [p.text for p in passages]
        embs = self._encode(texts)

        start_id = int(self.index.ntotal)
        self.index.add(embs)

        if self._meta_path is not None:
            with self._meta_path.open("a", encoding="utf-8") as f:
                for i, p in enumerate(passages):
                    rec = {
                        "vector_id": start_id + i,
                        "docid": p.docid,
                        "passage_idx": int(p.passage_idx),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return start_id, len(passages)

    def save(self, out_dir: str | Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = out_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(faiss_path))

    def open_metadata_log(self, out_dir: str | Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = out_dir / "meta.jsonl"
        # create if doesn't exist
        if not self._meta_path.exists():
            self._meta_path.write_text("", encoding="utf-8")

    def load(self, out_dir: str | Path):
        out_dir = Path(out_dir)
        self.index = faiss.read_index(str(out_dir / "faiss_index.bin"))
        # metadata is left as a file; you can read it on demand

def iter_hits_from_pyserini(
    engine,
    *,
    batch_size: int = 32,
    state_path: Path,
) -> Iterable[tuple[list[Hit], BuildState]]:
    lucene_reader = engine.reader.reader
    num_docs = int(engine.reader.stats()["documents"])
    print(f"num_docs: {num_docs}")
    state = _load_state(state_path)
    batch: list[Hit] = []

    for docnum in tqdm(range(state.next_docnum, num_docs), "ENCODING&INDEXING..."):
        try:
            d = lucene_reader.document(docnum)
            docid = d.get("id") or d.get("docid") or d.get("docno") or str(docnum)
            doc = engine.searcher.doc(docid)
            raw_doc = doc.raw()
            cleaned_doc, doc_metadata = clean_robust(raw_doc)
            batch.append(Hit(docid=str(docid), score=0.0, text=cleaned_doc))

            if len(batch) >= batch_size:
                yield batch, state
                batch = []

            state.next_docnum = docnum + 1

        except Exception as e:
            print(f"ERROR {e}")
            state.next_docnum = docnum + 1
            continue

    if batch:
        yield batch, state


def build_dense_faiss_from_pyserini(
    engine,
    embedding_function,
    out_dir: str | Path,
    *,
    hit_batch_size: int = 32,
    passage_batch_size: int = 64,
    save_every=50000
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "state.json"
    faiss_path = out_dir / "faiss_index.bin"

    vdb = VdbFaiss(embedding_function)
    vdb.open_metadata_log(out_dir)

    if faiss_path.exists():
        vdb.load(out_dir)

    state = _load_state(state_path)

    for i, (hit_batch, state) in enumerate(iter_hits_from_pyserini(
        engine,
        batch_size=hit_batch_size,
        state_path=state_path,
    )):
        try:
            passages = hits_to_passages(hit_batch)

            # chunk passages if splitter returns many
            for i in range(0, len(passages), passage_batch_size):
                chunk = passages[i:i + passage_batch_size]
                vdb.add_passages(chunk)

            if i%save_every==0:
                print(f"[INFO] batch {i} CHECKPOINTING")
                faiss.write_index(vdb.index, str(faiss_path))
                _save_state(state_path, state)

        except Exception as e:
            # hard safety: save and continue
            faiss.write_index(vdb.index, str(faiss_path))
            _save_state(state_path, state)
            print(f"[WARN] batch failed: {e}")
            continue

    # final checkpoint
    faiss.write_index(vdb.index, str(faiss_path))
    _save_state(state_path, state)



def load_vectorid_to_docid(meta_path: str):
    docids = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docids.append(rec["docid"])
    return docids


def search_vdb(query, index, encoder, meta_path, k, max_weight=0.8):
    def _collated_doc_score(scores, max_weight=0.8):
        return max(scores) * max_weight + (1 - max_weight) * (sum(scores) - max(scores)) / (len(scores) - 1) if len(
            scores) > 1 else max(scores)
    vectorid_to_docid = load_vectorid_to_docid(
        f"{meta_path}/meta.jsonl"
    )
    q_emb = encoder.encode([query]).astype("float32")
    scores, vector_ids = index.search(q_emb, k)
    scores = scores[0]
    vector_ids = vector_ids[0]
    retrieved_docs = [vectorid_to_docid[vid] for vid in vector_ids]
    per_doc_scores = defaultdict(list)
    for score, doc in zip(scores, retrieved_docs):
        per_doc_scores[doc].append(score)
    collated_doc_scores = {}
    for docid, scores in per_doc_scores.items():
        collated_doc_scores[docid] = _collated_doc_score(scores, max_weight=max_weight)
    ranked = sorted(collated_doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [Hit(docid=doc_score_pair[0], score=doc_score_pair[1]) for doc_score_pair in ranked]





def main():
    # Path to Pyserini Lucene index (ROBUST04)
    se = SearchEngine()

    # Output directory for FAISS + metadata + state
    OUT_DIR = "Nottogit/dense_faiss_robust04"

    # Encoder
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    build_dense_faiss_from_pyserini(
        engine=se,
        embedding_function=model,
        out_dir=OUT_DIR,
        hit_batch_size=32,        # how many docs go into splitter at once
        passage_batch_size=64,    # how many passages get embedded at once
    )
    print(torch.cuda.is_available())


if __name__ == "__main__":
    main()