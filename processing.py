import os
import ast
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, Union

load_dotenv()
SPLITTER_ARGS = ast.literal_eval(os.getenv("SPLITTER_ARGS"))
SPLITTER_SINGLETON = RecursiveCharacterTextSplitter(**SPLITTER_ARGS)

# Generic SGML-ish block: <TAG ...> ... </TAG>
# TAG names: letters/digits/_/-
_BLOCK = re.compile(
    r"<(?P<tag>[A-Za-z][A-Za-z0-9_-]*)(?P<attrs>\s+[^>]*)?>\s*(?P<content>.*?)\s*</(?P=tag)>",
    re.DOTALL
)

# Remove any remaining tags (inline or otherwise)
_ANY_TAG = re.compile(r"</?[^>]+>")

# Common “annotation-like” remnants you may want to drop from body
_TEXT_MARKER = re.compile(r"^\s*\[Text\]\s*", re.IGNORECASE)

@dataclass(slots=True)
class Hit:
    docid: str
    score: float
    query: Optional[str] = None
    text: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str):
        try:
            return self.meta[name]
        except KeyError:
            raise AttributeError(name)

    # For LangChain
    @property
    def page_content(self) -> str:
        return self.text

    @property
    def metadata(self) -> dict:
        return {"docid": self.docid, "query": self.query, **self.meta}

def _normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces/tabs
    s = re.sub(r"[ \t]+", " ", s)
    # Collapse many blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def clean_inner_text(s: str) -> str:
    """
    Cleans text inside a tag:
    - strips any nested tags like <F P=105> ... </F>
    - removes [Text] marker (common in newswire)
    - normalizes whitespace
    """
    s = _ANY_TAG.sub("", s)  # drop nested tags
    s = _TEXT_MARKER.sub("", s)  # drop leading [Text] marker
    return _normalize_ws(s)

def clean_robust(raw) -> Tuple[str, Dict[str, Union[str, List[str]]]]:
    """
    Extract ALL SGML-ish blocks.
      - <TEXT> blocks become the main body (concatenate if multiple)
      - every other tag becomes metadata[tag] (string or list of strings)
    Anything not inside blocks is ignored by default
    """
    metadata: Dict[str, Any] = {}
    body_parts: List[str] = []
    if not raw:
        return "", metadata

    # Find all blocks
    for m in _BLOCK.finditer(raw):
        tag = m.group("tag").strip().upper()
        content = m.group("content") or ""
        cleaned = clean_inner_text(content)

        if not cleaned:
            continue

        if tag == "TEXT":
            body_parts.append(cleaned)
        else:
            # store possibly repeated tags as list
            if tag in metadata:
                if isinstance(metadata[tag], list):
                    metadata[tag].append(cleaned)
                else:
                    metadata[tag] = [metadata[tag], cleaned]
            else:
                metadata[tag] = cleaned

    # If there was no <TEXT> tag, fall back to cleaning the whole raw as body
    # (useful when some corpora omit TEXT)
    if not body_parts:
        # Remove all blocks completely, then clean what remains
        stripped = _BLOCK.sub("", raw)
        stripped = clean_inner_text(stripped)
        return stripped, metadata

    body = "\n\n".join(body_parts)
    return body, metadata

def split_passages(hits: List[Hit], splitter=SPLITTER_SINGLETON):
    return splitter.split_documents(hits)
