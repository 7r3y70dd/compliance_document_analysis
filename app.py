# # app.py
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
#
# from models import AnalyzeRequest, AnalyzeResponse, ClauseMatch, MatchAlt
# from clause_extractor import extract_clauses
# from matcher import ClauseMatcher
# from settings import settings
# from comparator import numeric_downgrade
# from fastapi.staticfiles import StaticFiles
# from starlette.responses import RedirectResponse
# # NEW: NLI judge
# from nli_judge import NLIJudge, stricter  # requires nli_judge.py
# # NEW: LLM judge (classifies non_existent/partial as garbage/partial/satisfied + rationale)
# from llm_judge import LLMJudge  # requires llm_judge.py
#
# app = FastAPI(title="Policy Compliance Analyzer", version="1.4.0")
#
# app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
#
# # CORS for local dev; tighten in prod
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# matcher = ClauseMatcher()
#
# # Optional NLI annotate/downgrade
# nli = NLIJudge(settings.nli_model_name) if settings.use_nli_judge else None
#
# # Optional LLM refine (runs on partial + non_existent to upgrade/downgrade/mark garbage)
# # Use either USE_LLM_REFINE or USE_LLM_JUDGE to enable, whichever you’ve set.
# _use_llm = bool(getattr(settings, "use_llm_refine", False) or getattr(settings, "use_llm_judge", False))
# llm = (
#     LLMJudge(
#         model_name=settings.llm_model_name,
#         device=settings.llm_device,
#         hf_token=(settings.hf_hub_token or None),
#         max_new_tokens=settings.llm_max_new_tokens,
#     )
#     if _use_llm
#     else None
# )
#
# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "embedding_model": settings.embedding_model_name,
#         "use_cross_encoder": settings.use_cross_encoder,
#         "use_llm_judge": settings.use_llm_judge,
#         "use_llm_refine": getattr(settings, "use_llm_refine", False),
#         "use_semantic_normalizer": settings.use_semantic_normalizer,
#         "use_numeric_check": settings.use_numeric_check,
#         "use_nli_judge": settings.use_nli_judge,
#         "nli_model": settings.nli_model_name if settings.use_nli_judge else None,
#         "nli_annotate_only": settings.nli_annotate_only if settings.use_nli_judge else None,
#         "llm_model": settings.llm_model_name if _use_llm else None,
#     }
#
# @app.post("/analyze", response_model=AnalyzeResponse)
# def analyze(req: AnalyzeRequest):
#     # 1) size guard
#     if len(req.policy_text) > settings.max_chars or len(req.compliance_text) > settings.max_chars:
#         raise HTTPException(status_code=413, detail="Document too large")
#
#     # 2) clause extraction
#     policy_clauses = extract_clauses(req.policy_text)
#     compliance_clauses = extract_clauses(req.compliance_text)
#     if not policy_clauses or not compliance_clauses:
#         raise HTTPException(status_code=400, detail="Could not extract clauses from one or both documents")
#
#     # 3) retrieve matches
#     pairs = matcher.best_matches(compliance_clauses, policy_clauses, top_k=req.top_k)
#
#     # 4) scoring & labeling
#     matches = []
#     # NOTE: include 'garbage' now
#     counts = {"satisfied": 0, "partially_satisfied": 0, "non_existent": 0, "garbage": 0}
#
#     for i, alts in pairs:
#         best_idx, best_score = alts[0]
#         best_text = policy_clauses[best_idx]
#
#         # initial label from similarity (bi-encoder or cross-encoder score)
#         label = matcher.label_from_similarity(best_score)
#         rationale = None
#
#         # numeric downgrade first (e.g., 24h vs 48h)
#         if settings.use_numeric_check and label in {"satisfied", "partially_satisfied"}:
#             try:
#                 downgrade, reason = numeric_downgrade(compliance_clauses[i], best_text)
#                 if downgrade:
#                     label = "partially_satisfied"
#                     rationale = reason
#             except Exception:
#                 pass
#
#         # NLI judge (directional: policy should entail compliance)
#         if nli is not None:
#             try:
#                 nli_label, nli_scores = nli.classify(best_text, compliance_clauses[i])
#                 nli_note = (
#                     f"NLI verdict={nli_label} "
#                     f"(ent={nli_scores['entailment']:.2f}, neu={nli_scores['neutral']:.2f}, con={nli_scores['contradiction']:.2f})"
#                 )
#                 rationale = (rationale + " | " if rationale else "") + nli_note
#
#                 # If not annotate-only, constrain label toward stricter outcome (never upgrades)
#                 if not settings.nli_annotate_only:
#                     label = stricter(label, nli_label)
#             except Exception:
#                 pass
#
#         # LLM refine: ONLY run for partial + non_existent (your requirement)
#         # This can upgrade to satisfied/partial or mark as 'garbage' if unrelated.
#             # --- BEFORE: you call LLM on partial/non_existent unconditionally ---
#
#             # --- AFTER: gated + fast + always keep rationale ---
#             should_try_llm = (
#                     llm is not None
#                     and (
#                             (label == "non_existent" and settings.llm_on_non_existent) or
#                             (label == "partially_satisfied" and settings.llm_on_partial)
#                     )
#                     and float(best_score) >= settings.llm_min_sim_for_llm
#             )
#
#             if should_try_llm:
#                 try:
#                     alt_texts = [policy_clauses[j] for j, _ in alts[1:3]] if len(alts) > 1 else None
#                     res = llm.assess(
#                         requirement=compliance_clauses[i],
#                         best_policy=best_text,
#                         alt_snippets=alt_texts,
#                     )
#
#                     # Always append rationale if present (even if JSON parse failed)
#                     if res.get("rationale"):
#                         rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
#
#                     # Consider an upgrade/downgrade only if JSON parsed
#                     if res.get("_parsed"):
#                         proposed = (res.get("label") or "").strip().lower()
#
#                         # --- upgrade guardrails ---
#                         def can_upgrade(from_label: str, to_label: str) -> bool:
#                             # Block upgrades when sim is too low
#                             if float(best_score) < settings.llm_min_sim_for_upgrade:
#                                 return False
#                             # Optional: require NLI evidence (if NLI enabled and you've computed scores)
#                             # You have 'nli_scores' earlier in the loop when NLI is enabled
#                             if settings.use_nli_judge and settings.llm_require_nli_for_upgrade:
#                                 # You set nli_scores when you ran the NLI step; reuse here if available
#                                 try:
#                                     ent = nli_scores.get("entailment", 0.0)
#                                     if ent < settings.nli_entailment_min_for_upgrade:
#                                         return False
#                                 except Exception:
#                                     pass
#                             return True
#
#                         # Track original label and apply changes with guardrails
#                         original = label
#                         if proposed in {"satisfied", "partially_satisfied", "non_existent", "garbage"}:
#                             if proposed == "satisfied" and original != "satisfied":
#                                 if can_upgrade(original, proposed):
#                                     label = "satisfied"
#                             elif proposed == "partially_satisfied":
#                                 # partial is never stricter than non_existent? allow if guard passes
#                                 if original in {"non_existent", "garbage"} and can_upgrade(original, proposed):
#                                     label = "partially_satisfied"
#                                 elif original == "satisfied":
#                                     # allow downgrades freely
#                                     label = "partially_satisfied"
#                             elif proposed == "non_existent":
#                                 # allow downgrades freely
#                                 label = "non_existent"
#                             elif proposed == "garbage":
#                                 # allow downgrades freely
#                                 label = "garbage"
#
#                         # If upgraded/downgraded, add an "upgraded_from" field (see below)
#                         if label != original:
#                             upgraded_from = original
#                         else:
#                             upgraded_from = None
#
#                 except Exception:
#                     pass
#             else:
#                 upgraded_from = None
#
#             # VERSION 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#             #V2
#         # if llm is not None and label in {"partially_satisfied", "non_existent", "garbage"}:
#         #     try:
#         #         alt_texts = [policy_clauses[j] for j, _ in alts[1:3]]
#         #         res = llm.assess(
#         #             requirement=compliance_clauses[i],
#         #             best_policy=best_text,
#         #             alt_snippets=alt_texts,
#         #         )
#         #         #V2
#         #         if res.get("_parsed"):
#         #             llm_label = (res.get("label") or "").strip().lower()
#         #             if llm_label in {"satisfied", "partially_satisfied", "non_existent", "garbage"}:
#         #                 label = llm_label
#         #             if res.get("rationale"):
#         #                 rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
#         #         else:
#         #             if req.use_rationale and res.get("rationale"):
#         #                 rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
#         #
#         #         # AFTER (always keep the reason if present)
#         #         res = llm.assess(...)
#         #         if res.get("_parsed"):
#         #             llm_label = (res.get("label") or "").strip().lower()
#         #             if llm_label in {"satisfied", "partially_satisfied", "non_existent", "garbage"}:
#         #                 label = llm_label
#         #
#         #         # append rationale if provided, regardless of parse/use_rationale flag
#         #         if res.get("rationale"):
#         #             rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
#
#                 #V1
#                 # # Only trust the LLM if it produced valid JSON
#                 # if res.get("_parsed"):
#                 #     llm_label = (res.get("label") or "").strip().lower()
#                 #     if llm_label in {"satisfied", "partially_satisfied", "non_existent", "garbage"}:
#                 #         label = llm_label
#                 #     if res.get("rationale"):
#                 #         rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
#                 # else:
#                 #     # keep existing label; optional: attach note if user asked for rationale
#                 #     if req.use_rationale and res.get("rationale"):
#                 #         rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
#             # except Exception:
#             #     pass
#
#         #VERSION 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#         # Optional heuristic: ultra-low similarity means likely garbage
#         # (kept after LLM in case LLM is disabled or returns non_existent)
#         if label == "non_existent" and float(best_score) < 0.15:
#             label = "garbage"
#             if not rationale:
#                 rationale = "No meaningful semantic overlap with policy content."
#
#         counts[label] += 1
#
#         match = ClauseMatch(
#             id=f"C-{i:04d}",
#             text=compliance_clauses[i],
#             label=label,
#             best_match=MatchAlt(policy_text=best_text, similarity=round(best_score, 3)),
#             alternatives=[MatchAlt(policy_text=policy_clauses[j], similarity=round(s, 3)) for j, s in alts[1:]],
#             rationale=rationale if req.use_rationale or rationale else rationale,
#         )
#         # If your Pydantic model doesn't have upgraded_from, extend it; otherwise do it in dict form:
#         if hasattr(match, "upgraded_from") and upgraded_from:
#             match.upgraded_from = upgraded_from
#
#         matches.append(match)
#
#     return AnalyzeResponse(overall=counts, clauses=matches)
#
# @app.post("/analyze-multipart", response_model=AnalyzeResponse)
# async def analyze_multipart(
#     policy: UploadFile = File(...),
#     compliance: UploadFile = File(...),
#     top_k: int = Form(3),
#     use_rationale: bool = Form(False),
# ):
#     policy_text = (await policy.read()).decode("utf-8", errors="ignore")
#     compliance_text = (await compliance.read()).decode("utf-8", errors="ignore")
#     req = AnalyzeRequest(
#         policy_text=policy_text,
#         compliance_text=compliance_text,
#         top_k=top_k,
#         use_rationale=use_rationale,
#     )
#     return analyze(req)
#
# @app.get("/")
# def root():
#     return RedirectResponse(url="/ui")

# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import re

from models import AnalyzeRequest, AnalyzeResponse, ClauseMatch, MatchAlt
from clause_extractor import extract_clauses
from matcher import ClauseMatcher
from settings import settings
from comparator import numeric_downgrade
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
# NEW: NLI judge
from nli_judge import NLIJudge, stricter  # requires nli_judge.py
# NEW: LLM judge (classifies non_existent/partial as garbage/partial/satisfied + rationale)
from llm_judge import LLMJudge  # requires llm_judge.py

app = FastAPI(title="Policy Compliance Analyzer", version="1.4.0")

app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

# CORS for local dev; tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_REQ_HINTS = (
    "must","shall","required","requires","prohibit","ensure","encrypt","log","train",
    "review","report","notify","backup","restore","test","monitor","mfa","access",
    "control","policy","procedure","retain","limit","restrict","authenticate",
    "authorize","audit","incident","breach","privacy","security","safeguard",
    "disclose","agreement","contract","rotate","contingency","assign","officer"
)

def looks_requirement_like(t: str) -> bool:
    t = t.strip()
    tl = t.lower()
    if len(tl) >= 60 and (tl.endswith(".") or tl.endswith(":") or ";" in tl):
        return True
    if sum(ch.isdigit() for ch in tl) >= 2:
        return True
    if any(k in tl for k in _REQ_HINTS):
        return True
    # tags like (R), (A), (R/A) often indicate requirements in frameworks
    if re.search(r"\((?:r|a|r\/a|r\/?a)\)", tl):
        return True
    # short all-caps headings look like garbage:
    if len(t) < 40 and t.upper() == t and ":" not in t:
        return False
    return False


matcher = ClauseMatcher()

# Optional NLI annotate/downgrade
nli = NLIJudge(settings.nli_model_name) if settings.use_nli_judge else None

# Optional LLM refine (runs on partial + non_existent to upgrade/downgrade/mark garbage)
# Use either USE_LLM_REFINE or USE_LLM_JUDGE to enable, whichever you’ve set.
_use_llm = bool(getattr(settings, "use_llm_refine", False) or getattr(settings, "use_llm_judge", False))
llm = (
    LLMJudge(
        model_name=settings.llm_model_name,
        device=settings.llm_device,
        hf_token=(settings.hf_hub_token or None),
        max_new_tokens=settings.llm_max_new_tokens,
    )
    if _use_llm
    else None
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model_name,
        "use_cross_encoder": settings.use_cross_encoder,
        "use_llm_judge": settings.use_llm_judge,
        "use_llm_refine": getattr(settings, "use_llm_refine", False),
        "use_semantic_normalizer": settings.use_semantic_normalizer,
        "use_numeric_check": settings.use_numeric_check,
        "use_nli_judge": settings.use_nli_judge,
        "nli_model": settings.nli_model_name if settings.use_nli_judge else None,
        "nli_annotate_only": settings.nli_annotate_only if settings.use_nli_judge else None,
        "llm_model": settings.llm_model_name if _use_llm else None,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # 1) size guard
    if len(req.policy_text) > settings.max_chars or len(req.compliance_text) > settings.max_chars:
        raise HTTPException(status_code=413, detail="Document too large")

    # 2) clause extraction
    policy_clauses = extract_clauses(req.policy_text)
    compliance_clauses = extract_clauses(req.compliance_text)
    if not policy_clauses or not compliance_clauses:
        raise HTTPException(status_code=400, detail="Could not extract clauses from one or both documents")

    # 3) retrieve matches
    pairs = matcher.best_matches(compliance_clauses, policy_clauses, top_k=req.top_k)

    # 4) scoring & labeling
    matches = []
    # NOTE: include 'garbage' now
    counts = {"satisfied": 0, "partially_satisfied": 0, "non_existent": 0, "garbage": 0}

    for i, alts in pairs:
        best_idx, best_score = alts[0]
        best_text = policy_clauses[best_idx]

        # initial label from similarity (bi-encoder or cross-encoder score)
        label = matcher.label_from_similarity(best_score)
        rationale = None

        # numeric downgrade first (e.g., 24h vs 48h)
        if settings.use_numeric_check and label in {"satisfied", "partially_satisfied"}:
            try:
                downgrade, reason = numeric_downgrade(compliance_clauses[i], best_text)
                if downgrade:
                    label = "partially_satisfied"
                    rationale = reason
            except Exception:
                pass

        # NLI judge (directional: policy should entail compliance)
        if nli is not None:
            try:
                nli_label, nli_scores = nli.classify(best_text, compliance_clauses[i])
                nli_note = (
                    f"NLI verdict={nli_label} "
                    f"(ent={nli_scores['entailment']:.2f}, neu={nli_scores['neutral']:.2f}, con={nli_scores['contradiction']:.2f})"
                )
                rationale = (rationale + " | " if rationale else "") + nli_note

                # If not annotate-only, constrain label toward stricter outcome (never upgrades)
                if not settings.nli_annotate_only:
                    label = stricter(label, nli_label)
            except Exception:
                pass

            # LLM refine: ONLY run for partial + non_existent (your requirement)
            # This can upgrade to satisfied/partial or mark as 'garbage' if unrelated.
            # --- BEFORE: you call LLM on partial/non_existent unconditionally ---

            # --- AFTER: gated + fast + always keep rationale ---
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

                    # Always append rationale if present (even if JSON parse failed)
                    if res.get("rationale"):
                        rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"

                    # Consider an upgrade/downgrade only if JSON parsed
                    if res.get("_parsed"):
                        proposed = (res.get("label") or "").strip().lower()

                        # --- upgrade guardrails ---
                        def can_upgrade(from_label: str, to_label: str) -> bool:
                            # Block upgrades when sim is too low
                            if float(best_score) < settings.llm_min_sim_for_upgrade:
                                return False
                            # Optional: require NLI evidence (if NLI enabled and you've computed scores)
                            # You have 'nli_scores' earlier in the loop when NLI is enabled
                            if settings.use_nli_judge and settings.llm_require_nli_for_upgrade:
                                # You set nli_scores when you ran the NLI step; reuse here if available
                                try:
                                    ent = nli_scores.get("entailment", 0.0)
                                    if ent < settings.nli_entailment_min_for_upgrade:
                                        return False
                                except Exception:
                                    pass
                            return True

                        # Track original label and apply changes with guardrails
                        original = label
                        if proposed in {"satisfied", "partially_satisfied", "non_existent", "garbage"}:
                            if proposed == "satisfied" and original != "satisfied":
                                if can_upgrade(original, proposed):
                                    label = "satisfied"
                            elif proposed == "partially_satisfied":
                                # partial is never stricter than non_existent? allow if guard passes
                                if original in {"non_existent", "garbage"} and can_upgrade(original, proposed):
                                    label = "partially_satisfied"
                                elif original == "satisfied":
                                    # allow downgrades freely
                                    label = "partially_satisfied"
                            elif proposed == "non_existent":
                                # allow downgrades freely
                                label = "non_existent"
                            elif proposed == "garbage":
                                # allow downgrades freely
                                label = "garbage"

                        # If upgraded/downgraded, add an "upgraded_from" field (see below)
                        if label != original:
                            upgraded_from = original
                        else:
                            upgraded_from = None

                except Exception:
                    pass
            else:
                upgraded_from = None

            # VERSION 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # V2
        # if llm is not None and label in {"partially_satisfied", "non_existent", "garbage"}:
        #     try:
        #         alt_texts = [policy_clauses[j] for j, _ in alts[1:3]]
        #         res = llm.assess(
        #             requirement=compliance_clauses[i],
        #             best_policy=best_text,
        #             alt_snippets=alt_texts,
        #         )
        #         #V2
        #         if res.get("_parsed"):
        #             llm_label = (res.get("label") or "").strip().lower()
        #             if llm_label in {"satisfied", "partially_satisfied", "non_existent", "garbage"}:
        #                 label = llm_label
        #             if res.get("rationale"):
        #                 rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
        #         else:
        #             if req.use_rationale and res.get("rationale"):
        #                 rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
        #
        #         # AFTER (always keep the reason if present)
        #         res = llm.assess(...)
        #         if res.get("_parsed"):
        #             llm_label = (res.get("label") or "").strip().lower()
        #             if llm_label in {"satisfied", "partially_satisfied", "non_existent", "garbage"}:
        #                 label = llm_label
        #
        #         # append rationale if provided, regardless of parse/use_rationale flag
        #         if res.get("rationale"):
        #             rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"

        # V1
        # # Only trust the LLM if it produced valid JSON
        # if res.get("_parsed"):
        #     llm_label = (res.get("label") or "").strip().lower()
        #     if llm_label in {"satisfied", "partially_satisfied", "non_existent", "garbage"}:
        #         label = llm_label
        #     if res.get("rationale"):
        #         rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
        # else:
        #     # keep existing label; optional: attach note if user asked for rationale
        #     if req.use_rationale and res.get("rationale"):
        #         rationale = (rationale + " | " if rationale else "") + f"LLM: {res['rationale']}"
        # except Exception:
        #     pass

        # VERSION 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Optional heuristic: ultra-low similarity means likely garbage
        # (kept after LLM in case LLM is disabled or returns non_existent)
        # if label == "non_existent" and float(best_score) < 0.15:
        #     label = "garbage"
        #     if not rationale:
        #         rationale = "No meaningful semantic overlap with policy content."

        # BEFORE:
        # AFTER: only turn into garbage if it's not a real requirement
        GARBAGE_SIM_FLOOR = float(getattr(settings, "garbage_sim_floor", 0.08))  # optional env-driven
        if label == "non_existent" and float(best_score) < GARBAGE_SIM_FLOOR:
            if not looks_requirement_like(compliance_clauses[i]):
                label = "garbage"
                if not rationale:
                    rationale = "No meaningful semantic overlap; text appears to be a heading/boilerplate."
            else:
                # keep as non_existent because it reads like a valid requirement
                if not rationale:
                    rationale = "Valid requirement but no matching policy coverage found."

        counts[label] += 1

        match = ClauseMatch(
            id=f"C-{i:04d}",
            text=compliance_clauses[i],
            label=label,
            best_match=MatchAlt(policy_text=best_text, similarity=round(best_score, 3)),
            alternatives=[MatchAlt(policy_text=policy_clauses[j], similarity=round(s, 3)) for j, s in alts[1:]],
            rationale=rationale if req.use_rationale or rationale else rationale,
        )
        # If your Pydantic model doesn't have upgraded_from, extend it; otherwise do it in dict form:
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


@app.get("/")
def root():
    return RedirectResponse(url="/ui")