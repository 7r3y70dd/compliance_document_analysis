# clause_extractor.py  (REPLACE FILE)
import re
from typing import List

# Heuristics to skip headings/TOC/anchors/boilerplate
_HEADING_OR_META_HINTS = (
    "table of contents", "quick index", "mapping", "appendix", "raci",
    "contents", "index"
)
_JUNK_LINE_PATTERNS = (
    r'^\s*>{1,3}',          # blockquotes
    r'^\s*`{3,}',           # code fences
    r'^\s*<!--',            # HTML comments
)

_REQ_VERBS = (
    "must","shall","required","requires","prohibit","ensure","encrypt","log",
    "train","review","report","notify","backup","restore","test","monitor",
    "mfa","access","control","policy","procedure","retain","limit","restrict",
    "authenticate","authorize","audit","incident","breach","privacy",
    "security","safeguard","disclose","agreement","contract","rotate"
)

_HEADING_RE = re.compile(r'^\s{0,3}#{1,6}\s|^\s*[A-Z0-9][A-Z0-9\.\-\) ]{0,10}\s*$')
_LINK_MARK = re.compile(r'\[(.*?)\]\((.*?)\)')

def _is_heading_or_meta(line: str) -> bool:
    l = line.strip()
    if not l:
        return False
    if _HEADING_RE.search(l):
        return True
    if any(h in l.lower() for h in _HEADING_OR_META_HINTS):
        return True
    # TOC / anchors like "…](#section-id)"
    if "](" in l and "#" in l:
        # treat as meta unless clearly requirement-like and long enough
        if len(l) < 80:
            return True
    # All-caps short line (SECTION, PURPOSE, etc.)
    letters = [c for c in l if c.isalpha()]
    if letters and len(l) < 80 and sum(c.isupper() for c in letters)/len(letters) > 0.7:
        return True
    # Junk markers
    for pat in _JUNK_LINE_PATTERNS:
        if re.match(pat, l):
            return True
    return False

def _clean_line(line: str) -> str:
    # Strip bullets/numbering
    l = re.sub(r'^\s*[-*•]\s*', '', line)
    l = re.sub(r'^\s*(\d+[\.\)]\s*)+', '', l)
    # Unwrap markdown links
    l = _LINK_MARK.sub(r'\1', l)
    # Normalize spaces
    l = re.sub(r'\s+', ' ', l).strip()
    return l

def _looks_requirement_like(text: str) -> bool:
    t = text.lower()
    # Has numbers/units or timing → often a control/SLA
    if sum(ch.isdigit() for ch in t) >= 2:
        return True
    # Has any requirement-ish verb/keyword
    if any(v in t for v in _REQ_VERBS):
        return True
    # Otherwise, keep long, sentence-like statements
    return len(text) >= 60 and (text.endswith('.') or text.endswith(':') or ';' in text)

def extract_clauses(doc_text: str) -> List[str]:
    """Return requirement-like clauses, avoiding TOC/headers/anchors."""
    lines = doc_text.splitlines()
    clauses: List[str] = []
    buffer: List[str] = []

    def flush():
        nonlocal buffer
        if not buffer:
            return
        candidate = _clean_line(' '.join(buffer)).strip()
        buffer = []
        if not candidate:
            return
        if _looks_requirement_like(candidate):
            clauses.append(candidate)

    for raw in lines:
        line = raw.rstrip()

        if _is_heading_or_meta(line):
            flush()
            continue

        if not line.strip():
            flush()
            continue

        # Bullet lines: treat each bullet as a candidate clause
        if re.match(r'^\s*[-*•]\s+', line):
            flush()
            cleaned = _clean_line(line)
            if _looks_requirement_like(cleaned):
                clauses.append(cleaned)
            continue

        # Normal lines: accumulate until sentence end
        buffer.append(line.strip())
        if line.strip().endswith(('.', ';', ':')):
            flush()

    flush()

    # Deduplicate near-duplicates
    unique: List[str] = []
    seen = set()
    for c in clauses:
        key = re.sub(r'[^a-z0-9]+', ' ', c.lower())[:160]
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    return unique
