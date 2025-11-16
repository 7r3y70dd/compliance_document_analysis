# app.py
import os
import re
import json
import difflib
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging, time
from contextlib import contextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, PlainTextResponse
from pydantic import BaseModel

from models import AnalyzeRequest, AnalyzeResponse, ClauseMatch, MatchAlt
from clause_extractor import extract_clauses
from matcher import ClauseMatcher
from comparator import numeric_downgrade
from settings import settings
from nli_judge import NLIJudge, stricter
from llm_judge import LLMJudge

# NEW: editor/rewriter
from rewriter import EditorRewriter

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pca")

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    logger.info("⏳ %s ...", label)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info("✅ %s (%.2fs)", label, dt)


def _word_tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", s)

_REQ_HINTS = (
    "must","shall","required","requires","prohibit","ensure","encrypt","log","train",
    "review","report","notify","backup","restore","test","monitor","mfa","access",
    "control","policy","procedure","retain","limit","restrict","authenticate",
    "authorize","audit","incident","breach","privacy","security","safeguard",
    "disclose","agreement","contract","rotate","contingency","assign","officer"
)

_TITLE_STOPWORDS = {
    "scope","purpose","definitions","overview","summary","policy","procedure","procedures",
    "applicability","introduction","intent","objective","objectives","roles","responsibilities",
    "exceptions","exemptions","appendix","annex","glossary","references","revision","revisions",
    "enforcement","auditing","monitoring"
}

def _word_tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", s)

def is_title_like(t: str) -> bool:
    """
    Aggressively catch headings/titles.
    - very short (<= 6 tokens)
    - lacks sentence punctuation and looks TitleCase/ALLCAPS
    - outline-style like '1.', '1.2', 'A.', 'II ...' and still short
    """
    s = (t or "").strip()
    if not s:
        return True

    words = _word_tokens(s)
    n = len(words)
    if n <= 6:
        return True

    if not s.endswith((".", ":", ";")) and not any(p in s for p in (".", ";", ":")):
        is_all_caps = (s.upper() == s) and any(c.isalpha() for c in s)
        cap_words = sum(1 for w in words if w[:1].isupper() and (len(w) == 1 or w[1:].islower()))
        title_cased = (cap_words >= max(2, int(0.6 * n)))
        if is_all_caps or title_cased:
            return True

    if re.match(r"^\s*(\d+(\.\d+)*|[A-Z]|[IVXLC]+)\s*[\.\)]?\s+\S+", s):
        if n <= 8 and not s.rstrip().endswith("."):
            return True

    return False

_CLAUSE_BOILER_RE = re.compile(
    r'\*\*?Policy Clause:?\*\*?|'
    r'\*\*?\[End(?: of (?:Policy )?Clause)?\]\*?\*?|'
    r'—+\s*#+\s*Policy Clause:.*?—+\s*',
    re.IGNORECASE | re.DOTALL
)

def _sanitize_added_clause(s: str) -> str:
    """Make LLM-drafted additions parseable by our extractor."""
    s = (s or "").strip()
    # strip boilerplate wrappers and markdown bold
    s = _CLAUSE_BOILER_RE.sub("", s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.strip("*_`> ")
    # enforce sentence end; avoid trailing colons
    if not s.endswith((".", ";")):
        s += "."
    # keep it short & atomic (helps retrieval)
    if len(s) > 350:
        # split on '; ' or '. ' and keep first 2 small sentences
        parts = re.split(r';\s+|\.\s+', s)
        s = ". ".join([p.strip() for p in parts if p.strip()][:2]).rstrip(".") + "."
    return s

def looks_requirement_like(t: str) -> bool:
    """
    Opposite heuristic: real requirement lines.
    """
    t = (t or "").strip()
    tl = t.lower()
    if len(tl) >= 60 and (tl.endswith(".") or tl.endswith(":") or ";" in tl):
        return True
    if sum(ch.isdigit() for ch in tl) >= 2:
        return True
    # if any(k in tl for k in _REQ_HINTS):
    #     return True
    if re.search(r"\((?:r|a|r\/a|r\/?a)\)", tl):
        return True
    if len(t) < 40 and t.upper() == t and ":" not in t:
        return False
    return False

# -------------------- App init --------------------
app = FastAPI(title="Policy Compliance Analyzer", version="1.6.0")
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

matcher = ClauseMatcher()
nli = NLIJudge(settings.nli_model_name) if settings.use_nli_judge else None

_use_llm = bool(getattr(settings, "use_llm_refine", False) or getattr(settings, "use_llm_judge", False))
llm = (
    LLMJudge(
        model_name=settings.llm_model_name,
        device=settings.llm_device,
        hf_token=(settings.hf_hub_token or None),
        max_new_tokens=settings.llm_max_new_tokens,
    )
    if _use_llm else None
)

# Editor (rewriter) — can reuse LLM device by default
editor = EditorRewriter(
    model_name=os.getenv("EDITOR_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
    device=int(os.getenv("EDITOR_DEVICE", str(settings.llm_device))),
    max_new_tokens=int(os.getenv("EDITOR_MAX_NEW_TOKENS", "128")),
    load_in_4bit=os.getenv("EDITOR_LOAD_IN_4BIT", "1").lower() in ("1", "true", "yes"),
    hf_token=(settings.hf_hub_token or None),
)

# -------------------- Health --------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "app_version": "1.6.0",
        "embedding": {
            "model": settings.embedding_model_name,
            "device": settings.embedding_device,
            "use_e5_prefixes": settings.use_e5_prefixes,
        },
        "retrieval": {
            "use_cross_encoder": settings.use_cross_encoder,
            "cross_encoder_model": settings.cross_encoder_model_name,
            "cross_encoder_top_k": settings.cross_encoder_top_k,
            "top_k_default": settings.top_k_default,
            "round_similarity": settings.round_similarity,
        },
        "thresholds": {
            "satisfied_threshold": settings.satisfied_threshold,
            "partial_threshold": settings.partial_threshold,
        },
        "semantic": {
            "use_semantic_normalizer": settings.use_semantic_normalizer,
            "semantic_lexicon_path": settings.semantic_lexicon_path,
            "semantic_threshold": settings.semantic_threshold,
            "semantic_max_tags": settings.semantic_max_tags,
            "use_numeric_check": settings.use_numeric_check,
        },
        "nli": {
            "use_nli_judge": settings.use_nli_judge,
            "model": settings.nli_model_name if settings.use_nli_judge else None,
            "device": settings.nli_device if settings.use_nli_judge else None,
            "annotate_only": settings.nli_annotate_only if settings.use_nli_judge else None,
            "satisfied_floor": settings.nli_satisfied_floor if settings.use_nli_judge else None,
            "partial_floor": settings.nli_partial_floor if settings.use_nli_judge else None,
            "entailment_min": settings.nli_entailment_min if settings.use_nli_judge else None,
            "contradiction_min": settings.nli_contradiction_min if settings.use_nli_judge else None,
        },
        "llm": {
            "enabled": _use_llm,
            "use_llm_judge": settings.use_llm_judge,
            "use_llm_refine": getattr(settings, "use_llm_refine", False),
            "model": settings.llm_model_name if _use_llm else None,
            "device": settings.llm_device if _use_llm else None,
            "max_new_tokens": settings.llm_max_new_tokens if _use_llm else None,
        },
        "editor": {
            "model": os.getenv("EDITOR_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
            "device": int(os.getenv("EDITOR_DEVICE", str(settings.llm_device))),
            "max_new_tokens": int(os.getenv("EDITOR_MAX_NEW_TOKENS", "128")),
            "load_in_4bit": os.getenv("EDITOR_LOAD_IN_4BIT", "1").lower() in ("1","true","yes"),
        },
        "limits": {"max_chars": settings.max_chars},
    }

# -------------------- Analyze endpoints (unchanged) --------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if len(req.policy_text) > settings.max_chars or len(req.compliance_text) > settings.max_chars:
        raise HTTPException(status_code=413, detail="Document too large")

    policy_clauses = extract_clauses(req.policy_text)
    compliance_clauses = extract_clauses(req.compliance_text)
    if not policy_clauses or not compliance_clauses:
        raise HTTPException(status_code=400, detail="Could not extract clauses from one or both documents")

    pairs = matcher.best_matches(compliance_clauses, policy_clauses, top_k=req.top_k)

    matches = []
    counts = {"satisfied": 0, "partially_satisfied": 0, "non_existent": 0, "garbage": 0}

    for i, alts in pairs:
        best_idx, best_score = alts[0]
        best_text = policy_clauses[best_idx]

        label = matcher.label_from_similarity(best_score)
        rationale = None
        upgraded_from = None  # keep if you emit this field

        # numeric downgrade (if any)
        if settings.use_numeric_check and label in {"satisfied", "partially_satisfied"}:
            try:
                downgrade, reason = numeric_downgrade(compliance_clauses[i], best_text)
                if downgrade:
                    label = "partially_satisfied"
                    rationale = reason
            except Exception:
                pass

        # --- HARD HEADING FILTER: mark titles as garbage and SKIP NLI/LLM ---
        if is_title_like(compliance_clauses[i]):
            label = "garbage"
            rationale = "Short/heading-like text (no actionable requirement)."

            counts[label] += 1
            match = ClauseMatch(
                id=f"C-{i:04d}",
                text=compliance_clauses[i],
                label=label,
                best_match=MatchAlt(policy_text=best_text, similarity=round(float(best_score or 0.0), 3)),
                alternatives=[MatchAlt(policy_text=policy_clauses[j], similarity=round(s, 3)) for j, s in alts[1:]],
                rationale=rationale if req.use_rationale or rationale else rationale,
            )
            matches.append(match)
            continue  # <<< do NOT run NLI/LLM for headings

        # --- NLI (if enabled) ---
        if nli is not None:
            try:
                nli_label, nli_scores = nli.classify(best_text, compliance_clauses[i])
                nli_note = (
                    f"NLI verdict={nli_label} "
                    f"(ent={nli_scores['entailment']:.2f}, neu={nli_scores['neutral']:.2f}, con={nli_scores['contradiction']:.2f})"
                )
                rationale = (rationale + " | " if rationale else "") + nli_note
                if not settings.nli_annotate_only:
                    label = stricter(label, nli_label)
            except Exception:
                pass

        # --- LLM refine (only for partial/non_existent as you had) ---
        should_try_llm = (
            llm is not None
            and (
                (label == "non_existent" and settings.llm_on_non_existent) or
                (label == "partially_satisfied" and settings.llm_on_partial)
            )
            and float(best_score) >= settings.llm_min_sim_for_llm
        )

        if should_try_llm:
            try:
                alt_texts = [policy_clauses[j] for j, _ in alts[1:3]] if len(alts) > 1 else None
                res = llm.assess(
                    requirement=compliance_clauses[i],
                    best_policy=best_text,
                    alt_snippets=alt_texts,
                )
                if res.get("rationale"):
                    rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
                if res.get("_parsed"):
                    proposed = (res.get("label") or "").strip().lower()

                    def can_upgrade(from_label: str, to_label: str) -> bool:
                        if float(best_score) < settings.llm_min_sim_for_upgrade:
                            return False
                        if settings.use_nli_judge and settings.llm_require_nli_for_upgrade:
                            try:
                                if nli_scores.get("entailment", 0.0) < settings.nli_entailment_min_for_upgrade:
                                    return False
                            except Exception:
                                pass
                        return True

                    original = label
                    if proposed in {"satisfied","partially_satisfied","non_existent","garbage"}:
                        if proposed == "satisfied" and original != "satisfied":
                            if can_upgrade(original, proposed):
                                label = "satisfied"
                        elif proposed == "partially_satisfied":
                            if original in {"non_existent","garbage"} and can_upgrade(original, proposed):
                                label = "partially_satisfied"
                            elif original == "satisfied":
                                label = "partially_satisfied"
                        elif proposed == "non_existent":
                            label = "non_existent"
                        elif proposed == "garbage":
                            label = "garbage"
                    if label != original:
                        upgraded_from = original
            except Exception:
                pass

        # --- ultra-low sim: keep non_existent, but demote to garbage if it doesn't look like a requirement
        GARBAGE_SIM_FLOOR = float(getattr(settings, "garbage_sim_floor", 0.05))
        if label == "non_existent" and float(best_score) < GARBAGE_SIM_FLOOR:
            if not looks_requirement_like(compliance_clauses[i]):
                label = "garbage"
                if not rationale:
                    rationale = "No meaningful semantic overlap; text appears to be a heading/boilerplate."
            else:
                if not rationale:
                    rationale = "Valid requirement but no matching policy coverage found."

        counts[label] += 1
        match = ClauseMatch(
            id=f"C-{i:04d}",
            text=compliance_clauses[i],
            label=label,
            best_match=MatchAlt(policy_text=best_text, similarity=round(float(best_score or 0.0), 3)),
            alternatives=[MatchAlt(policy_text=policy_clauses[j], similarity=round(s, 3)) for j, s in alts[1:]],
            rationale=rationale if req.use_rationale or rationale else rationale,
        )
        if hasattr(match, "upgraded_from") and upgraded_from:
            match.upgraded_from = upgraded_from

        matches.append(match)

    return AnalyzeResponse(overall=counts, clauses=matches)

@app.post("/analyze-multipart", response_model=AnalyzeResponse)
async def analyze_multipart(
    policy: UploadFile = File(...),
    compliance: UploadFile = File(...),
    top_k: int = Form(3),
    use_rationale: bool = Form(False),
):
    policy_text = (await policy.read()).decode("utf-8", errors="ignore")
    compliance_text = (await compliance.read()).decode("utf-8", errors="ignore")
    req = AnalyzeRequest(
        policy_text=policy_text,
        compliance_text=compliance_text,
        top_k=top_k,
        use_rationale=use_rationale,
    )
    return analyze(req)

# -------------------- Rewrite Core --------------------
def _normalize(s: str) -> str:
    s = re.sub(r'\s+', ' ', s.strip())
    s = s.replace("’","'").replace("“","\"").replace("”","\"")
    return s

def _find_span(hay: str, needle: str) -> Optional[Tuple[int,int]]:
    idx = hay.find(needle)
    if idx != -1:
        return idx, idx + len(needle)
    H, N = _normalize(hay), _normalize(needle)
    if len(N) < 12:
        return None
    m = difflib.SequenceMatcher(None, H, N).find_longest_match(0, len(H), 0, len(N))
    if m.size / len(N) >= 0.8:
        j = hay.lower().find(needle.strip().lower()[: max(10, int(0.6*len(needle))) ])
        if j != -1:
            return j, j + len(needle)
    return None

def _rewrite_core(policy_text: str, analysis: AnalyzeResponse):
    PROGRESS_EVERY = settings.progress_every
    with timed("Rewriting policy from analysis JSON"):
        text = policy_text
        changes = []
        cache = {}
        new_clauses = []

        total = len(analysis.clauses)
        for n, c in enumerate(analysis.clauses, 1):
            if n == 1 or n % PROGRESS_EVERY == 0 or n == total:
                logger.info("… rewrite pass on clause %d/%d (%s)", n, total, c.id)
        # we only attempt to rewrite when there is a concrete best-match snippet
            if c.label not in {"partially_satisfied", "non_existent"}:
                continue
            if not c.best_match or not c.best_match.policy_text:
                continue

            requirement = c.text
            policy_clause = c.best_match.policy_text
            rationale = (c.rationale or "").strip()

            # If rationale indicates no coverage, plan to add a new clause at the bottom
            if re.search(r"valid requirement.*no matching policy coverage found\.?", rationale, re.IGNORECASE):
                try:
                    drafted = editor.draft_new_clause(requirement)
                    if drafted and drafted.strip():
                        new_clauses.append({"id": c.id, "requirement": requirement, "draft": drafted.strip()})
                except Exception:
                    pass  # don't break the run if drafting fails

            # Propose a precise edit for the current clause
            key = (requirement, policy_clause, rationale)
            proposal = cache.get(key) or editor.propose_edit(requirement, policy_clause, rationale)
            cache[key] = proposal

            if not proposal or "rewritten" not in proposal or "original" not in proposal:
                continue

            orig = proposal["original"].strip() or policy_clause
            rew  = proposal["rewritten"].strip() or policy_clause
            status = proposal.get("status", "rewritten")

            # Unchanged? log and continue
            if status == "unchanged" or _normalize(orig) == _normalize(rew):
                changes.append({
                    "id": c.id,
                    "requirement": requirement,
                    "original_snippet": policy_clause,
                    "rewritten_snippet": policy_clause,
                    "status": "unchanged",
                    "notes": proposal.get("notes",""),
                })
                continue

            # Locate and patch the original doc
            span = _find_span(text, policy_clause) or _find_span(text, orig)
            if not span:
                changes.append({
                    "id": c.id,
                    "requirement": requirement,
                    "original_snippet": policy_clause,
                    "rewritten_snippet": rew,
                    "status": "skipped_not_found",
                    "notes": "Could not locate original snippet safely.",
                })
                continue

            a, b = span
            text = text[:a] + rew + text[b:]
            changes.append({
                "id": c.id,
                "requirement": requirement,
                "original_snippet": policy_clause,
                "rewritten_snippet": rew,
                "status": "rewritten",
                "notes": proposal.get("notes",""),
            })

    # Append newly drafted clauses at bottom
    if new_clauses:
        appendix_header = "\n\nAdditional Controls (Auto-Added):\n"
        lines = []
        for i, nc in enumerate(new_clauses, 1):
            clean = _sanitize_added_clause(nc["draft"])
            lines.append(f"{i}. {clean}")
            changes.append({
                "id": nc["id"],
                "requirement": nc["requirement"],
                "original_snippet": "",
                "rewritten_snippet": clean,
                "status": "added_new_clause",
                "notes": "No coverage in policy; clause appended at document end."
            })
        text = text.rstrip() + appendix_header + "\n".join(lines) + "\n"

    return text, changes

# -------------------- Read analysis JSON from folder + rewrite --------------------
JSON_DIR = Path("./resources/json_results").resolve()
POLICY_DIR = Path("./resources/company_policy").resolve()
OUT_DIR = (JSON_DIR / "out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

class RewriteResponse(BaseModel):
    rewritten_policy_text: str
    change_log: list

def _load_analysis_json(path: Path) -> AnalyzeResponse:
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data.get("analysis", data)  # support either raw or wrapped
    return AnalyzeResponse(**payload)

def _load_policy_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

@app.post("/rewrite-from-json", response_model=RewriteResponse)
def rewrite_from_json(
    result: str = Query(..., description="File under ./resources/json_results (e.g., my_run.json)"),
    policy: str = Query(..., description="File under ./resources/company_policy (e.g., mid_generated_policy.txt)"),
):
    result_path = (JSON_DIR / result).resolve()
    policy_path = (POLICY_DIR / policy).resolve()
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Result JSON not found: {result_path}")
    if not policy_path.exists():
        raise HTTPException(status_code=404, detail=f"Policy text not found: {policy_path}")

    analysis = _load_analysis_json(result_path)
    policy_text = _load_policy_text(policy_path)

    rewritten, logs = _rewrite_core(policy_text, analysis)
    return RewriteResponse(rewritten_policy_text=rewritten, change_log=logs)

@app.post("/rewrite-txt-from-json")
def rewrite_txt_from_json(
    result: str = Query(..., description="File under ./resources/json_results"),
    policy: str = Query(..., description="File under ./resources/company_policy"),
):
    result_path = (JSON_DIR / result).resolve()
    policy_path = (POLICY_DIR / policy).resolve()
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Result JSON not found: {result_path}")
    if not policy_path.exists():
        raise HTTPException(status_code=404, detail=f"Policy text not found: {policy_path}")

    analysis = _load_analysis_json(result_path)
    policy_text = _load_policy_text(policy_path)

    rewritten, _ = _rewrite_core(policy_text, analysis)
    out_name = f"{result_path.stem}__rewritten.txt"
    headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
    return PlainTextResponse(rewritten, headers=headers)

@app.post("/rewrite-folder")
def rewrite_folder(
    policy: str = Query(..., description="Policy filename (under ./resources/company_policy) used for all JSONs"),
):
    policy_path = (POLICY_DIR / policy).resolve()
    if not policy_path.exists():
        raise HTTPException(status_code=404, detail=f"Policy text not found: {policy_path}")
    policy_text = _load_policy_text(policy_path)

    processed: List[Dict] = []
    for path in sorted(JSON_DIR.glob("*.json")):
        try:
            analysis = _load_analysis_json(path)
            rewritten, _ = _rewrite_core(policy_text, analysis)
            out_path = OUT_DIR / f"{path.stem}__rewritten.txt"
            out_path.write_text(rewritten, encoding="utf-8")
            processed.append({"input": str(path), "output": str(out_path), "status": "ok"})
        except Exception as e:
            processed.append({"input": str(path), "status": "error", "error": str(e)})
    return {"folder": str(JSON_DIR), "policy": str(policy_path), "results": processed}

@app.get("/")
def root():
    return RedirectResponse(url="/ui")
