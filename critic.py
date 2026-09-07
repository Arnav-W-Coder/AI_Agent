"""
critic.py — Claim-level grounding check and surgical answer repair.

The critic is intentionally limited to two possible LLM calls per query:
  1. check() after generation
  2. repair() only when unsupported claims are found

Polishing is deterministic so it cannot introduce new facts after grounding.
"""
import logging
import re
from typing import Optional, TYPE_CHECKING

from langchain_ollama import ChatOllama

if TYPE_CHECKING:
    from config import RAGConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

log = logging.getLogger(__name__)

UNCERTAINTY_PHRASES = [
    "i don't have enough information",
    "i don't know",
    "i cannot find",
    "i was unable to find",
    "the documents do not",
    "the context does not",
    "no information",
    "not mentioned",
    "not present in",
    "cannot answer",
    "not provided",
    "insufficient information",
    "unable to determine",
    "no relevant",
]

_CRITIC_PROMPT = ChatPromptTemplate.from_template("""
Check the ANSWER for factual claims that are not supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

A claim is supported when the context states it directly or supports it by a
clear synonym, paraphrase, or logically equivalent wording. Do not require
identical wording. Omission is not hallucination.

If every factual claim is supported, output exactly:
FULLY_GROUNDED

Otherwise output exactly:
UNSUPPORTED_CLAIMS:
• <the unsupported claim>
• <the unsupported claim>

Output only that result. No explanation or reasoning.
""")

_REPAIR_PROMPT = ChatPromptTemplate.from_template("""
Repair the draft using only the retrieved context.

RETRIEVED CONTEXT:
{context}

DRAFT ANSWER:
{answer}

UNSUPPORTED CLAIMS:
{claims}

Rules:
- Remove unsupported claims or rewrite only the sentence(s) containing them so
  the sentence becomes fully supported by the context.
- Preserve all supported information and the answer's useful structure.
- You may rephrase a flagged sentence when needed; do not preserve inaccurate
  wording merely to keep it verbatim.
- Do not add facts, examples, explanations, or citations that are not supported
  by the context.
- Return only the corrected answer.
- If no meaningful supported answer remains, return exactly:
  I don't have enough information to answer this confidently.
""")


class CriticAndRepair:
    """Claim-level hallucination detection with one optional repair pass."""

    def __init__(self, llm: ChatOllama, cfg: Optional["RAGConfig"] = None) -> None:
        self._critic_chain = _CRITIC_PROMPT | llm | StrOutputParser()
        self._repair_chain = _REPAIR_PROMPT | llm | StrOutputParser()
        self._cfg = cfg

    def check(self, context: str, answer: str) -> tuple[str, str, float]:
        if not context or not context.strip():
            log.info("[Critic] No context — HALLUCINATED")
            return "HALLUCINATED", "No retrieval context was provided.", 0.0

        if self._is_uncertainty_response(answer):
            log.info("[Critic] Uncertainty admission — auto GROUNDED")
            return "GROUNDED", "", 1.0

        raw = self._critic_chain.invoke({"context": context, "answer": answer}).strip()
        return self._parse_critic_output(raw)

    def repair(self, context: str, answer: str, claims: str) -> str:
        log.info("[Repair] Removing/reworking %d flagged claim(s)...", claims.count("•"))
        repaired = self._repair_chain.invoke({
            "context": context,
            "answer": answer,
            "claims": claims,
        }).strip()
        if not repaired or len(repaired) < 5:
            log.warning("[Repair] Empty repair — using grounded fallback")
            return "I don't have enough information to answer this confidently."
        log.info("[Repair] Done. Length: %d → %d chars", len(answer), len(repaired))
        return repaired

    def polish(self, answer: str) -> str:
        """Deterministically remove a small set of meta-hedges; no new LLM call."""
        if self._is_uncertainty_response(answer):
            return "The available sources do not contain enough information to answer this question."
        patterns = [
            r"\bBased on (?:the )?(?:provided|retrieved) (?:context|sources),?\s*",
            r"\bAccording to (?:the )?(?:provided|retrieved) (?:context|sources),?\s*",
            r"\bThe context suggests that\s*",
            r"\bIt appears that\s*",
            r"\bIt seems that\s*",
        ]
        cleaned = answer
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip() or answer

    def compute_faithfulness(self, verdict: str, claims: str) -> float:
        if verdict == "GROUNDED":
            return 1.0
        penalty = self._cfg.critic_claim_penalty if self._cfg is not None else 0.20
        n_claims = max(1, claims.count("•"))
        return max(0.0, round(1.0 - penalty * n_claims, 2))

    def _is_uncertainty_response(self, answer: str) -> bool:
        sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
        if not sentences:
            return False
        uncertainty_count = sum(
            1 for sentence in sentences
            if any(phrase in sentence.lower() for phrase in UNCERTAINTY_PHRASES)
        )
        threshold = self._cfg.critic_uncertainty_threshold if self._cfg is not None else 0.50
        ratio = uncertainty_count / len(sentences)
        return ratio >= threshold

    def _parse_critic_output(self, raw: str) -> tuple[str, str, float]:
        upper = raw.upper()
        if "FULLY_GROUNDED" in upper or "FULLY GROUNDED" in upper:
            return "GROUNDED", "", 1.0
        if "UNSUPPORTED_CLAIMS" in upper or "UNSUPPORTED CLAIMS" in upper:
            claims_section = re.sub(
                r"UNSUPPORTED[_ ]CLAIMS\s*:?\s*", "", raw, flags=re.IGNORECASE
            ).strip()
            n = max(1, claims_section.count("•"))
            penalty = self._cfg.critic_claim_penalty if self._cfg is not None else 0.20
            score = max(0.0, round(1.0 - penalty * n, 2))
            return "HALLUCINATED", claims_section, score
        log.warning("[Critic] Unexpected response format: '%s' — defaulting GROUNDED", raw[:80])
        return "GROUNDED", "", 0.8
