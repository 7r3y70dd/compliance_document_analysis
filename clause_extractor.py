import re
from typing import List

_heading_re = re.compile(r"^\s*(?:[A-Z][\w\s]{2,}|[\w\s]{2,}):\s*$")
_bullet_re = re.compile(r"^\s*(?:[-*•\u2022]|\d+\.|[a-z]\))\s+")

def _normalize(text: str) -> str:
    # Normalize whitespace and strip noise.
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\u00a0", " ", text)
    return text.strip()

def split_into_blocks(text: str) -> List[str]:
    text = _normalize(text)
    lines = text.split("\n")
    blocks, cur = [], []
    for ln in lines:
        if _heading_re.match(ln) and cur:
            blocks.append(" ".join(cur).strip())
            cur = [ln.strip()]
        else:
            cur.append(ln.strip())
    if cur:
        blocks.append(" ".join(cur).strip())
    # Remove tiny blocks
    return [b for b in blocks if len(b.split()) >= 3]

def split_bullets(block: str) -> List[str]:
    parts, cur = [], []
    for ln in block.split("\n"):
        if _bullet_re.match(ln) and cur:
            parts.append(" ".join(cur).strip())
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        parts.append(" ".join(cur).strip())
    # If no bullets detected, return the block itself
    if len(parts) == 1:
        return parts
    return [p for p in parts if len(p.split()) >= 3]

def fallback_sentence_split(block: str) -> List[str]:
    # Conservative split on end punctuation.
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", block)
    return [s.strip() for s in sents if len(s.split()) >= 5]

def extract_clauses(text: str) -> List[str]:
    clauses: List[str] = []
    for blk in split_into_blocks(text):
        bul = split_bullets(blk)
        if len(bul) <= 2:  # not really a bulleted list
            clauses.extend(fallback_sentence_split(blk))
        else:
            clauses.extend(bul)
    # Deduplicate (rough)
    seen = set()
    uniq = []
    for c in clauses:
        key = re.sub(r"\W+", " ", c.lower()).strip()
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq