# entity_store.py
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from tqdm import tqdm


@dataclass
class EntityStore:
    """
    Fast random access to entities per docid using a JSONL + offsets index.

    JSONL line format (per doc):
      {"docid": "LA010190-0016", "entities": [["Samuel Beckett","PERSON"], ...]}

    offsets.pkl stores:
      offsets[docid] = byte_offset_in_jsonl_file
    """

    jsonl_path: Path
    offsets: Dict[str, int]

    # DF stats over the *subset* file (computed once)
    N: int = 0
    entity_df: Optional[Dict[str, int]] = None

    @classmethod
    def load(cls, jsonl_path: str, offsets_path: str, build_df: bool = True) -> "EntityStore":
        jsonl_p = Path(jsonl_path)
        off_p = Path(offsets_path)

        with off_p.open("rb") as f:
            offsets = pickle.load(f)

        store = cls(jsonl_path=jsonl_p, offsets=offsets)

        if build_df:
            store.build_entity_df()

        return store

    def get(self, docid: str) -> List[Tuple[str, str]]:
        """
        Return list of (entity_text, label) for docid.
        If not present, returns empty list.
        """
        off = self.offsets.get(docid)
        if off is None:
            return []

        with self.jsonl_path.open("rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8", errors="replace")

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return []

        ents = obj.get("entities", [])
        out: List[Tuple[str, str]] = []
        for item in ents:
            if not isinstance(item, list) or len(item) != 2:
                continue
            out.append((str(item[0]), str(item[1])))
        return out

    def build_entity_df(self, progress_every: int = 5000) -> None:
        """
        Compute entity document frequency over the *subset* of docs included
        in this JSONL file (i.e., only docids present in offsets).

        DF(entity) = #docs in subset that contain entity at least once (case-insensitive).
        """
        df = defaultdict(int)
        docids = list(self.offsets.keys())
        self.N = len(docids)

        # Iterate through offsets docids and read each line once
        # (random seeks; still fine for a subset size like ~30k-70k)
        # If you want faster sequential reading, tell me and I’ll give that version too.
        with self.jsonl_path.open("rb") as f:
            for i, docid in enumerate(tqdm(docids, desc="[entity_df] building DF over subset")):
                off = self.offsets[docid]
                f.seek(off)
                line = f.readline().decode("utf-8", errors="replace")

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ents = obj.get("entities", [])
                seen = set()
                for item in ents:
                    if not isinstance(item, list) or len(item) != 2:
                        continue
                    ent_text = str(item[0]).strip().lower()
                    if not ent_text:
                        continue
                    if ent_text in seen:
                        continue
                    seen.add(ent_text)

                for ent in seen:
                    df[ent] += 1

                if progress_every and (i + 1) % progress_every == 0:
                    tqdm.write(f"[entity_df] processed {i+1}/{self.N} docs... unique_ents={len(df)}")

        self.entity_df = dict(df)

    def idf(self, entity_lower: str) -> float:
        """
        IDF-like penalty computed on subset statistics:
          idf = log((N + 1) / (df + 1))

        If DF stats are not built, returns 1.0 (no penalty).
        """
        if self.entity_df is None or self.N <= 0:
            return 1.0
        df = self.entity_df.get(entity_lower, 0)
        return math.log((self.N + 1.0) / (df + 1.0))
