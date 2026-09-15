"""
critic.py — Multi-dimensional RAG quality control and bounded repair.
"""
import logging
import re
from typing import Literal, Optional, TYPE_CHECKING, TypedDict

from langchain_ollama import ChatOllama

if TYPE_CHECKING:
    from config import RAGConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from checkpoints import checkpoint

log = logging.getLogger(__name__)

# Single source of truth for the "couldn't answer" text. pipeline.py imports
# this instead of keeping its own copy, so the two files can't drift apart —
# this was previously duplicated as a separate literal in pipeline.py, and a
# third, different wording lived in this file's own polish() method.
CANONICAL_INSUFFICIENT_INFO_RESPONSE = (
    "I don't have enough information to answer this confidently."
)

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

_DESTRUCTIVE_REPAIR_PROMPT = ChatPromptTemplate.from_template("""
Regenerate an answer after a hallucination was detected. Use ONLY the
retrieved context and the original question.

You may completely restructure the previous answer, but every factual
statement must be directly supported by the context. Do not use outside
knowledge or preserve unsupported claims. If the context is insufficient,
return exactly: I don't have enough information to answer this confidently.

Original question: {question}
Previous answer: {answer}
Critic findings: {issues}
Retrieved context: {context}

Return only the answer.
""")


class CriticResult(TypedDict):
    verdict: Literal["PASS", "HALLUCINATED", "UNCERTAIN"]
    severity: Literal["none", "minor", "major"]
    unsupported_claims: list[str]
    repairable: bool
    critique: str
    repaired_answer: str | None


class CriticAndRepair:
    """Multi-dimensional evaluation with one bounded repair pass."""

    def __init__(self, llm: ChatOllama, cfg: Optional["RAGConfig"] = None) -> None:
        self._critic_chain = _CRITIC_PROMPT | llm | StrOutputParser()
        self._repair_chain = _REPAIR_PROMPT | llm | StrOutputParser()
        self._destructive_repair_chain = _DESTRUCTIVE_REPAIR_PROMPT | llm | StrOutputParser()
        self._cfg = cfg
        self.last_details: dict = {}

    def check(self, *args) -> tuple[str, str, float]:
        if len(args) == 3:
            question, context, answer = args
        elif len(args) == 2:
            question = ""
            context, answer = args
        else:
            raise TypeError("check() expects (question, context, answer) or (context, answer)")

        result = self.evaluate(question, answer, context)
        score = 1.0 if result["verdict"] == "PASS" else 0.0
        legacy_verdict = "GROUNDED" if result["verdict"] == "PASS" else result["verdict"]
        return legacy_verdict, result["critique"], score

    def evaluate(self, question: str, answer: str, context: str) -> CriticResult:
        checkpoint("critic.input", {"question": question, "context": context, "answer": answer},
                   enabled=getattr(self._cfg, "debug_checkpoints", True),
                   preview_chars=getattr(self._cfg, "checkpoint_preview_chars", 160),
                   sample_items=getattr(self._cfg, "checkpoint_sample_items", 3))
        if not context or not context.strip():
            return self._result("HALLUCINATED", "major", "No retrieval context was provided.", False)
        if self._is_uncertainty_response(answer):
            return self._result("PASS", "none", "", True)
        try:
            raw = self._critic_chain.invoke({
                "question": question or "Determine whether the answer is supported by the retrieved context.",
                "context": context,
                "answer": answer,
            }).strip()
            result = self._parse_critic_output(raw)
            checkpoint("critic.evaluation", result, enabled=getattr(self._cfg, "debug_checkpoints", True),
                       preview_chars=getattr(self._cfg, "checkpoint_preview_chars", 160),
                       sample_items=getattr(self._cfg, "checkpoint_sample_items", 3))
            return result
        except Exception as exc:
            log.warning("[Critic] Evaluation failed: %s", exc)
            return self._result("UNCERTAIN", "none", f"Critic evaluation failed: {exc}", False)

    def repair_constrained(self, question: str, context: str, answer: str,
                           critique: str) -> str | None:
        return self._run_repair(self._repair_chain, question, context, answer, critique)

    def repair_destructive(self, question: str, context: str, answer: str,
                           critique: str) -> str | None:
        return self._run_repair(self._destructive_repair_chain, question, context, answer, critique)

    def _run_repair(self, chain, question: str, context: str, answer: str, critique: str) -> str | None:
        try:
            repaired = chain.invoke({"question": question, "context": context,
                                     "answer": answer, "issues": critique}).strip()
            repaired = self._strip_meta_commentary(repaired)
            return repaired if len(repaired) >= 5 else None
        except Exception as exc:
            log.warning("[Repair] Failed: %s", exc)
            return None

    def _result(self, verdict: Literal["PASS", "HALLUCINATED", "UNCERTAIN"],
                severity: Literal["none", "minor", "major"], critique: str,
                repairable: bool) -> CriticResult:
        # These three call sites (no context, self-admitted uncertainty, and
        # an evaluation call that raised) never run the three-dimension
        # critic prompt, so completeness/relevance were never actually
        # judged. Report that honestly as "N/A" instead of omitting the keys
        # — pipeline.py's metrics were previously papering over the gap with
        # a ".get(..., 'UNKNOWN')" default, which looked like a failure to
        # evaluate rather than the intentional short-circuit it is.
        if verdict == "UNCERTAIN":
            # The evaluation itself didn't complete, so groundedness is
            # unknown too — reporting "FAIL" here would claim a determination
            # that was never made.
            groundedness = "N/A"
        else:
            groundedness = "PASS" if verdict == "PASS" else "FAIL"
        self.last_details = {
            "groundedness": groundedness, "completeness": "N/A", "relevance": "N/A",
        }
        return {"verdict": verdict, "severity": severity, "unsupported_claims": [critique] if critique else [],
                "repairable": repairable, "critique": critique, "repaired_answer": None}

    def repair(self, *args) -> str:
        if len(args) == 4:
            question, context, answer, issues = args
        elif len(args) == 3:
            question = ""
            context, answer, issues = args
        else:
            raise TypeError("repair() expects (question, context, answer, issues) or (context, answer, issues)")

        checkpoint("critic.repair_input", {
            "question": question, "context": context, "answer": answer, "issues": issues,
        }, enabled=getattr(self._cfg, "debug_checkpoints", True),
                   preview_chars=getattr(self._cfg, "checkpoint_preview_chars", 160),
                   sample_items=getattr(self._cfg, "checkpoint_sample_items", 3))
        log.info("[Repair] Reworking flagged issue(s)...")
        repaired = self._repair_chain.invoke({
            "question": question or "Answer the user's question only from the supplied context.",
            "context": context, "answer": answer, "issues": issues,
        }).strip()
        repaired = self._strip_meta_commentary(repaired)
        if not repaired or len(repaired) < 5:
            log.warning("[Repair] Empty repair — using grounded fallback")
            return CANONICAL_INSUFFICIENT_INFO_RESPONSE
        checkpoint("critic.repair_output", repaired, enabled=getattr(self._cfg, "debug_checkpoints", True),
                   preview_chars=getattr(self._cfg, "checkpoint_preview_chars", 160),
                   sample_items=getattr(self._cfg, "checkpoint_sample_items", 3),
                   chars_before=len(answer), chars_after=len(repaired))
        log.info("[Repair] Done. Length: %d → %d chars", len(answer), len(repaired))
        return repaired

    def polish(self, answer: str) -> str:
        if self._is_uncertainty_response(answer):
            result = CANONICAL_INSUFFICIENT_INFO_RESPONSE
        else:
            patterns = [
                r"\bBased on (?:the )?(?:provided|retrieved) (?:context|sources),?\s*",
                r"\bAccording to (?:the )?(?:provided|retrieved) (?:context|sources),?\s*",
                r"\bThe context suggests that\s*", r"\bIt appears that\s*", r"\bIt seems that\s*",
            ]
            result = answer
            for pattern in patterns:
                result = re.sub(pattern, "", result, flags=re.IGNORECASE)
            result = self._strip_meta_commentary(result).strip() or answer
        checkpoint("critic.polish_output", result, enabled=getattr(self._cfg, "debug_checkpoints", True),
                   preview_chars=getattr(self._cfg, "checkpoint_preview_chars", 160),
                   sample_items=getattr(self._cfg, "checkpoint_sample_items", 3))
        return result

    def compute_faithfulness(self, verdict: str, claims: str) -> float:
        if verdict == "GROUNDED":
            return 1.0
        penalty = getattr(self._cfg, "critic_claim_penalty", 0.20)
        n_claims = max(1, claims.count("\n- ") or claims.count("•"))
        return max(0.0, round(1.0 - penalty * n_claims, 2))

    def _is_uncertainty_response(self, answer: str) -> bool:
        sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
        if not sentences:
            return False
        uncertainty_count = sum(1 for sentence in sentences if any(phrase in sentence.lower() for phrase in UNCERTAINTY_PHRASES))
        threshold = getattr(self._cfg, "critic_uncertainty_threshold", 0.50)
        return uncertainty_count / len(sentences) >= threshold

    def _parse_critic_output(self, raw: str) -> CriticResult:
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
        self.last_details = {"groundedness": groundedness, "completeness": completeness, "relevance": relevance}
        failed = [value for value in self.last_details.values() if value == "FAIL"]
        if not failed:
            verdict: Literal["PASS", "HALLUCINATED", "UNCERTAIN"] = "PASS"
            severity: Literal["none", "minor", "major"] = "none"
        else:
            verdict = "HALLUCINATED"
            severity = "major" if groundedness == "FAIL" else "minor"
        return {
            "verdict": verdict,
            "severity": severity,
            "unsupported_claims": [line.strip("- ") for line in issues.splitlines() if line.strip()],
            "repairable": bool(issues),
            "critique": issues,
            "repaired_answer": None,
        }

    @staticmethod
    def _strip_meta_commentary(answer: str) -> str:
        if not answer:
            return answer
        lines = answer.splitlines()
        kept = []
        for line in lines:
            low = line.strip().lower()
            if (low.startswith("note: i removed")
                or low.startswith("note: no additional information was added")
                or low.startswith("i removed the unsupported")
                or low.startswith("the answer was repaired")
                or low.startswith("i preserved the supported")
                or low.startswith("i reworked the unsupported")
                or low.startswith("no additional information was added")):
                continue
            kept.append(line)
        return "\n".join(kept).strip()