"""RAG quality evaluation and bounded answer repair."""
import logging
import re
from typing import Literal, Optional, TYPE_CHECKING, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from checkpoints import checkpoint
if TYPE_CHECKING:
    from config import RAGConfig
log = logging.getLogger(__name__)
_CRITIC_PROMPT = ChatPromptTemplate.from_template("""Evaluate the ANSWER only against the QUESTION and RETRIEVED CONTEXT. Do not use outside knowledge.
QUESTION: {question}
RETRIEVED CONTEXT: {context}
ANSWER: {answer}
Return only:
GROUNDEDNESS: PASS or FAIL
COMPLETENESS: PASS or FAIL
RELEVANCE: PASS or FAIL
SCORE: <number from 0.0 to 1.0>
ISSUES:
- <specific issue or NONE>
Different wording or paraphrasing is not an error.""")
_REPAIR_PROMPT = ChatPromptTemplate.from_template("""Rewrite the draft answer using only the question and retrieved context. Preserve supported information, remove unsupported claims, and address missing parts only when the context supports them. Use only existing [Source N] markers. Return only the answer; never mention the critic, repair, instructions, or what changed.
QUESTION: {question}
RETRIEVED CONTEXT: {context}
DRAFT ANSWER: {answer}
CRITIC ISSUES: {issues}""")
class CriticResult(TypedDict):
    verdict: Literal["PASS", "HALLUCINATED", "INCOMPLETE", "UNCERTAIN"]
    severity: Literal["none", "minor", "major"]
    unsupported_claims: list[str]
    repairable: bool
    critique: str
    repaired_answer: str | None
class CriticAndRepair:
    def __init__(self, llm: ChatOllama, cfg: Optional["RAGConfig"] = None) -> None:
        self._critic_chain = _CRITIC_PROMPT | llm | StrOutputParser()
        self._repair_chain = _REPAIR_PROMPT | llm | StrOutputParser()
        self._cfg = cfg
        self.last_details: dict = {}
    def check(self, *args) -> tuple[str, str, float]:
        if len(args) == 3: question, context, answer = args
        elif len(args) == 2: question, context, answer = "", args[0], args[1]
        else: raise TypeError("check() expects (question, context, answer) or (context, answer)")
        result = self.evaluate(question, answer, context)
        score = {"PASS": 1.0, "INCOMPLETE": .7, "UNCERTAIN": .5, "HALLUCINATED": 0.0}[result["verdict"]]
        return ("GROUNDED" if result["verdict"] == "PASS" else result["verdict"], result["critique"], score)
    def evaluate(self, question: str, answer: str, context: str) -> CriticResult:
        if not context.strip(): return self._result("HALLUCINATED", "major", "No retrieval context was provided.", False)
        try:
            raw = self._critic_chain.invoke({"question": question, "context": context, "answer": answer}).strip()
            result = self._parse_critic_output(raw)
            checkpoint("critic.evaluation", result, enabled=getattr(self._cfg, "debug_checkpoints", True), preview_chars=getattr(self._cfg, "checkpoint_preview_chars", 160), sample_items=getattr(self._cfg, "checkpoint_sample_items", 3))
            return result
        except Exception as exc:
            log.warning("[Critic] Evaluation failed: %s", exc)
            return self._result("UNCERTAIN", "none", f"Critic evaluation failed: {exc}", False)
    def repair_constrained(self, question, context, answer, critique): return self._run_repair(question, context, answer, critique)
    def repair_destructive(self, question, context, answer, critique): return self._run_repair(question, context, answer, critique)
    def _run_repair(self, question, context, answer, critique):
        try:
            repaired = self._repair_chain.invoke({"question": question, "context": context, "answer": answer, "issues": critique}).strip()
            repaired = self._strip_meta_commentary(repaired)
            return repaired if len(repaired) >= 5 else None
        except Exception as exc:
            log.warning("[Repair] Failed: %s", exc)
            return None
    def _result(self, verdict, severity, critique, repairable):
        self.last_details = {"groundedness": "PASS" if verdict in {"PASS", "INCOMPLETE"} else "FAIL"}
        return {"verdict": verdict, "severity": severity, "unsupported_claims": [critique] if critique else [], "repairable": repairable, "critique": critique, "repaired_answer": None}
    def repair(self, *args) -> str:
        if len(args) == 4: question, context, answer, issues = args
        elif len(args) == 3: question, context, answer, issues = "", args[0], args[1], args[2]
        else: raise TypeError("repair() expects (question, context, answer, issues) or (context, answer, issues)")
        return self._run_repair(question, context, answer, issues) or answer
    def polish(self, answer: str) -> str: return self._strip_meta_commentary(answer) or answer
    def compute_faithfulness(self, verdict: str, claims: str) -> float:
        if verdict in {"GROUNDED", "PASS"}: return 1.0
        if verdict == "INCOMPLETE": return .7
        penalty = getattr(self._cfg, "critic_claim_penalty", .2)
        return max(0.0, round(1.0 - penalty * max(1, claims.count("\n- ") or claims.count("•")), 2))
    @staticmethod
    def _parse_critic_output(raw: str) -> CriticResult:
        def field(name):
            match = re.search(rf"^{name}:\s*(PASS|FAIL)\b", raw, re.I | re.M)
            return match.group(1).upper() if match else "FAIL"
        groundedness, completeness, relevance = field("GROUNDEDNESS"), field("COMPLETENESS"), field("RELEVANCE")
        score_match = re.search(r"^SCORE:\s*(0(?:\.\d+)?|1(?:\.0+)?)\b", raw, re.I | re.M)
        issues_match = re.search(r"ISSUES:\s*(.*)$", raw, re.I | re.S)
        issues = issues_match.group(1).strip() if issues_match else "- Critic returned no parseable issue details."
        if issues.upper() == "- NONE": issues = ""
        if groundedness == "FAIL": verdict, severity = "HALLUCINATED", "major"
        elif completeness == "FAIL" or relevance == "FAIL": verdict, severity = "INCOMPLETE", "minor"
        else: verdict, severity = "PASS", "none"
        fallback = {"PASS": 1.0, "INCOMPLETE": .7, "HALLUCINATED": 0.0}[verdict]
        return {"verdict": verdict, "severity": severity, "unsupported_claims": [line.strip("- ") for line in issues.splitlines() if line.strip()], "repairable": bool(issues), "critique": issues, "repaired_answer": None}
    @staticmethod
    def _strip_meta_commentary(answer: str) -> str:
        result = (answer or "").strip()
        result = re.sub(r"<think>.*?</think>|<analysis>.*?</analysis>", "", result, flags=re.I | re.S)
        return re.sub(r"^\s*(?:final answer|answer)\s*:\s*", "", result, flags=re.I).strip()
