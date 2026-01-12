# build_offsets.py
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict


def build_offsets(jsonl_path: str, out_path: str | None = None) -> str:
    jsonl = Path(jsonl_path)
    if out_path is None:
        out_path = str(jsonl.with_suffix(jsonl.suffix + ".offsets.pkl"))

    offsets: Dict[str, int] = {}
    with jsonl.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                obj = json.loads(line.decode("utf-8"))
                docid = obj.get("docid")
                if docid:
                    offsets[docid] = pos
            except Exception:
                continue

    with open(out_path, "wb") as g:
        pickle.dump(offsets, g, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path
