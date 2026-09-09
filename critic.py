"""
critic.py — Multi-dimensional RAG quality control and bounded repair.

The critic evaluates a generated answer across three dimensions:
  1. groundedness: are factual claims supported by retrieved context?
  2. completeness: does the answer address the user's question using available evidence?
  3. relevance: is the retrieved context actually useful for the question?

Document relevance is primarily established before generation by the retrieval
CrossEncoder. The LLM critic then performs answer-level groundedness and
completeness evaluation in one call. Repair is bounded to avoid runaway loops.
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
You are a strict RAG quality evaluator. Evaluate the ANSWER against the
QUESTION and RETRIEVED CONTEXT. Do not use outside knowledge.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

Evaluate three dimensions:
1. GROUNDEDNESS: Every factual claim in the answer must be directly supported
   by the context or by a clear synonym, paraphrase, or logically equivalent
   statement. Do not require identical wording. Omission is not hallucination.
2. COMPLETENESS: The answer should directly address the question and cover the
   important parts that the context actually supports. Do not penalize the
   answer for information that the context does not contain.
3. RELEVANCE: The retrieved context must contain useful evidence for answering
   the question. Judge the context itself, not whether the answer is eloquent.

Return ONLY this format:
GROUNDEDNESS: PASS or FAIL
COMPLETENESS: PASS or FAIL
RELEVANCE: PASS or FAIL
SCORE: <number from 0.0 to 1.0>
ISSUES:
- <specific issue, if any>
- <specific issue, if any>

If there are no issues, write:
ISSUES:
- NONE

Be conservative but fair. A claim is not hallucinated merely because the
context uses different wording.
""")

_REPAIR_PROMPT = ChatPromptTemplate.from_template("""
Repair the draft answer using ONLY the retrieved context and the question.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

DRAFT ANSWER:
{answer}

CRITIC ISSUES:
{issues}

Rules:
- Preserve every supported, useful part of the draft whenever possible.
- Remove unsupported factual claims or rewrite them so they are fully
  supported by the context.
- Address missing parts of the question only when the context supports them.
- Do not add outside knowledge, examples, numbers, citations, or terminology.
- If the context cannot support a requested point, state that briefly rather
  than inventing an answer.
- Return ONLY the corrected answer.
- Never mention the critic, repair, flagged claims, grounding, context quality,
  what you changed, or this instruction.
- Never append a note, disclaimer, or explanation about the repair.
- If no meaningful supported answer remains, return exactly:
I don't have enough information to answer this confidently.
""")


class CriticAndRepair:
    """Multi-dimensional RAG evaluation with bounded answer repair."""

    def __init__(self, llm: ChatOllama, cfg: Optional["RAGConfig"] = None) -> None:
        self._critic_chain = _CRITIC_PROMPT | llm | StrOutputParser()
        self._repair_chain = _REPAIR_PROMPT | llm | StrOutputParser()
        self._cfg = cfg

    def check(self, question: str, context: str, answer: str) -> tuple[str, str, float, dict]:
        """Evaluate groundedness, completeness, and relevance in one LLM call."""
        if not context or not context.strip():
            log.info("[Critic] No context — FAIL")
            return "HALLUCINATED", "No retrieval context was provided.", 0.0, {
                "groundedness": "FAIL", "completeness": "FAIL", "relevance": "FAIL"
            }

        if self._is_uncertainty_response(answer):
            return "GROUNDED", "", 1.0, {
                "groundedness": "PASS", "completeness": "PASS", "relevance": "PASS"
            }

        raw = self._critic_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
        }).strip()
        return self._parse_critic_output(raw)

    def repair(self, question: str, context: str, answer: str, issues: str) -> str:
        issue_count = max(0, issues.count("\n- "))
        log.info("[Repair] Reworking %d flagged issue(s)...", issue_count)
        repaired = self._repair_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
            "issues": issues,
        }).strip()
        repaired = self._strip_meta_commentary(repaired)
        if not repaired or len(repaired) < 5:
            log.warning("[Repair] Empty repair — using grounded fallback")
            return "I don't have enough information to answer this confidently."
        log.info("[Repair] Done. Length: %d → %d chars", len(answer), len(repaired))
        return repaired

    def polish(self, answer: str) -> str:
        """Deterministically remove meta-hedges without introducing facts."""
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
        cleaned = self._strip_meta_commentary(cleaned)
        return cleaned.strip() or answer

    def compute_faithfulness(self, verdict: str, claims: str) -> float:
        """Compatibility helper; the critic's explicit score is preferred."""
        if verdict == "GROUNDED":
            return 1.0
        penalty = self._cfg.critic_claim_penalty if self._cfg is not None else 0.20
        n_claims = max(1, claims.count("\n- ") or claims.count("•"))
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
        return uncertainty_count / len(sentences) >= threshold

    def _parse_critic_output(self, raw: str) -> tuple[str, str, float, dict]:
        def field(name: str, default: str = "FAIL") -> str:
            match = re.search(rf"^{name}:\s*(PASS|FAIL)\b", raw, flags=re.IGNORECASE | re.MULTILINE)
            return match.group(1).upper() if match else default

        groundedness = field("GROUNDEDNESS")
        completeness = field("COMPLETENESS")
        relevance = field("RELEVANCE")
        score_match = re.search(r"^SCORE:\s*(0(?:\.\d+)?|1(?:\.0+)?)\b", raw, flags=re.IGNORECASE | re.MULTILINE)
        score = float(score_match.group(1)) if score_match else (1.0 if groundedness == "PASS" else 0.0)

        issues_match = re.search(r"ISSUES:\s*(.*)$", raw, flags=re.IGNORECASE | re.DOTALL)
        issues = issues_match.group(1).strip() if issues_match else "- Critic returned no parseable issue details."
        if issues.upper() == "- NONE":
            issues = ""

        details = {
            "groundedness": groundedness,
            "completeness": completeness,
            "relevance": relevance,
        }
        failed = [name for name, value in details.items() if value == "FAIL"]
        verdict = "GROUNDED" if not failed else "HALLUCINATED"
        if not raw.strip():
            return "HALLUCINATED", "- Empty critic response.", 0.0, {
                "groundedness": "FAIL", "completeness": "FAIL", "relevance": "FAIL"
            }
        return verdict, issues, round(max(0.0, min(1.0, score)), 2), details

    @staticmethod
    def _strip_meta_commentary(answer: str) -> str:
        """Remove common LLM repair commentary that must never reach the user."""
        if not answer:
            return answer
        lines = answer.splitlines()
        kept = []
        for line in lines:
            low = line.strip().lower()
            if (
                low.startswith("note: i removed")
                or low.startswith("note: no additional information was added")
                or low.startswith("i removed the unsupported")
                or low.startswith("the answer was repaired")
                or low.startswith("i preserved the supported")
                or low.startswith("i reworked the unsupported")
            ):
                continue
            kept.append(line)
        return "\n".join(kept).strip()
