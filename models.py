# models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class AnalyzeRequest(BaseModel):
    policy_text: str = Field(..., description="Full company policy text")
    compliance_text: str = Field(..., description="Full compliance document text")
    top_k: int = Field(3, ge=1, le=10, description="Number of alternative matches to include")
    use_rationale: bool = Field(False, description="If true, generate natural-language rationales (slower)")

class MatchAlt(BaseModel):
    policy_text: str
    similarity: float

class ClauseMatch(BaseModel):
    id: str
    text: str

    label: str
    best_match: Optional[MatchAlt]
    alternatives: List[MatchAlt] = []
    rationale: Optional[str] = None

class AnalyzeResponse(BaseModel):

    overall: dict
    clauses: List[ClauseMatch]
