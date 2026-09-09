"""
critic.py — Multi-dimensional RAG quality control and bounded repair.

The critic evaluates a generated answer across three dimensions:
  1. groundedness: are factual claims supported by retrieved context?
  2. completeness: does the answer address the user's question using available evidence?
  3. relevance: is the retrieved context actually useful for the question?

The public methods remain backward-compatible with the existing pipeline while
supporting the richer question-aware interface for future callers.
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
    "i don't have enough information", "i don't know", "i cannot find",
    "i was unable to find", "the documents do not", "the context does not",
    "no information", "not mentioned", "not present in", "cannot answer",
    "not provided", "insufficient information", "unable to determine", "no relevant",
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
1. GROUNDEDNESS: Every factual claim must be directly supported by the context
   or by a clear synonym, paraphrase, or logically equivalent statement.
2. COMPLETENESS: The answer should directly address the question and cover the
   important parts that the context actually supports. Do not penalize missing
   information that the context itself does not contain.
3. RELEVANCE: The retrieved context must contain useful evidence for the
   question. Judge the context itself, not writing quality.

Return ONLY:
GROUNDEDNESS: PASS or FAIL
COMPLETENESS: PASS or FAIL
RELEVANCE: PASS or FAIL
SCORE: <number from 0.0 to 1.0>
ISSUES:
- <specific issue, if any>

If there are no issues, write:
ISSUES:
- NONE

Be conservative but fair. Different wording alone is not hallucination.
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
- Preserve supported, useful information whenever possible.
- Remove unsupported factual claims or rewrite them so they are fully supported.
- Address missing parts only when the context supports them.
- Do not add outside knowledge, examples, numbers, citations, or terminology.
- If the context cannot support a requested point, say so briefly rather than inventing it.
- Return ONLY the corrected answer.
- Never mention the critic, repair, flagged claims, grounding, context quality,
  what you changed, or these instructions.
- Never append a note, disclaimer, or explanation about the repair.
- If no meaningful supported answer remains, return exactly:
I don't have enough information to answer this confidently.
""")


class CriticAndRepair:
    """Multi-dimensional evaluation with one bounded repair pass."""

    def __init__(self, llm: ChatOllama, cfg: Optional["RAGConfig"] = None) -> None:
        self._critic_chain = _CRITIC_PROMPT | llm | StrOutputParser()
        self._repair_chain = _REPAIR_PROMPT | llm | StrOutputParser()
        self._cfg = cfg
        self.last_details: dict = {}

    def check(self, *args) -> tuple[str, str, float]:
        """Evaluate relevance, groundedness, and completeness.

        Supports both check(question, context, answer) and the legacy
        check(context, answer) signature used by the current pipeline.
        """
        if len(args) == 3:
            question, context, answer = args
        elif len(args) == 2:
            question = ""
            context, answer = args
        else:
            raise TypeError("check() expects (question, context, answer) or (context, answer)")

        if not context or not context.strip():
            log.info("[Critic] No context — FAIL")
            self.last_details = {"groundedness": "FAIL", "completeness": "FAIL", "relevance": "FAIL"}
            return "HALLUCINATED", "- No retrieval context was provided.", 0.0

        if self._is_uncertainty_response(answer):
            self.last_details = {"groundedness": "PASS", "completeness": "PASS", "relevance": "PASS"}
            return "GROUNDED", "", 1.0

        raw = self._critic_chain.invoke({
            "question": question or "Determine whether the answer is supported by the retrieved context.",
            "context": context,
            "answer": answer,
        }).strip()
        return self._parse_critic_output(raw)

    def repair(self, *args) -> str:
        """Repair using either the new or legacy argument order."""
        if len(args) == 4:
            question, context, answer, issues = args
        elif len(args) == 3:
            question = ""
            context, answer, issues = args
        else:
            raise TypeError("repair() expects (question, context, answer, issues) or (context, answer, issues)")

        log.info("[Repair] Reworking flagged issue(s)...")
        repaired = self._repair_chain.invoke({
            "question": question or "Answer the user's question only from the supplied context.",
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
        """Deterministically remove meta-commentary and hedges."""
        if self._is_uncertainty_response(answer):
            return "The available sources do not contain enough information to answer this question."
        patterns = [
            r"\bBased on (?:the )?(?:provided|retrieved) (?:context|sources),?\s*",
            r"\bAccording to (?:the )?(?:provided|retrieved) (?:context|sources),?\s*",
            r"\bThe context suggests that\s*", r"\bIt appears that\s*",
            r"\bIt seems that\s*",
        ]
        cleaned = answer
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = self._strip_meta_commentary(cleaned)
        return cleaned.strip() or answer

    def compute_faithfulness(self, verdict: str, claims: str) -> float:
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

    def _parse_critic_output(self, raw: str) -> tuple[str, str, float]:
        def field(name: str, default: str = "FAIL") -> str:
            match = re.search(
                rf"^{name}:\s*(PASS|FAIL)\b", raw,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            return match.group(1).upper() if match else default

        groundedness = field("GROUNDEDNESS")
        completeness = field("COMPLETENESS")
        relevance = field("RELEVANCE")
        score_match = re.search(
            r"^SCORE:\s*(0(?:\.\d+)?|1(?:\.0+)?)\b", raw,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        score = float(score_match.group(1)) if score_match else (1.0 if groundedness == "PASS" else 0.0)

        issues_match = re.search(r"ISSUES:\s*(.*)$", raw, flags=re.IGNORECASE | re.DOTALL)
        issues = issues_match.group(1).strip() if issues_match else "- Critic returned no parseable issue details."
        if issues.upper() == "- NONE":
            issues = ""

        self.last_details = {
            "groundedness": groundedness,
            "completeness": completeness,
            "relevance": relevance,
        }
        failed = [value for value in self.last_details.values() if value == "FAIL"]
        verdict = "GROUNDED" if not failed else "HALLUCINATED"
        return verdict, issues, round(max(0.0, min(1.0, score)), 2)

    @staticmethod
    def _strip_meta_commentary(answer: str) -> str:
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
                or low.startswith("no additional information was added")
            ):
                continue
            kept.append(line)
        return "\n".join(kept).strip()
