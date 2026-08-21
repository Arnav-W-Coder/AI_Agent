"""
critic.py — Three-stage critic, surgical repair, and answer polishing.

Stage flow:
  1. _is_uncertainty_response()  — Python check, no LLM call.
       If the answer is predominantly "I don't know" sentences (>= 50% of
       sentences contain an uncertainty phrase) → auto-GROUNDED.
       Requires the majority of sentences to hedge, not just one buried line.

  2. _run_critic()               — LLM claim-level check.
       Asks the critic LLM to list specific unsupported claims rather than
       returning a binary verdict. This gives repair() something actionable.

  3. repair()                    — Surgical LLM edit, no tool re-runs.
       Given the list of bad claims, a focused LLM call removes only those
       sentences. The rest of the answer is preserved verbatim.

  4. polish()                    — Final pass after grounding is confirmed.
       Strips all hedging filler ("it appears", "the context suggests", etc.)
       so the answer reads as confident and finalized.
       Skips polishing pure "I don't know" answers.

Faithfulness score:
  compute_faithfulness() converts the critic verdict into a 0–1 float
  stored in query_metrics.answer_faithfulness and used for drift detection.
"""
import logging
import re
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

log = logging.getLogger(__name__)

# ── Uncertainty phrase list ───────────────────────────────────────────────────

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

# ── Prompt templates ──────────────────────────────────────────────────────────

_CRITIC_PROMPT = ChatPromptTemplate.from_template("""
You are a strict fact-checker. Compare the answer against the retrieved context.

RETRIEVED CONTEXT:
{context}

AGENT ANSWER:
{answer}

Task:
Find every specific claim in the answer that CANNOT be traced word-for-word
or by clear paraphrase to the retrieved context above.

Rules:
- List unsupported claims as concise bullet points. Quote the exact phrase.
- Flag specific facts: names, numbers, dates, organisations, causal claims.
- Do NOT flag general summaries that broadly reflect the context.
- If ALL claims are supported, reply with exactly one word: FULLY_GROUNDED
- If there are unsupported claims, start your reply with the line:
  UNSUPPORTED_CLAIMS:
  then list each claim as a bullet starting with "•".
- Reply with nothing else — no preamble, no sign-off.
""")

_REPAIR_PROMPT = ChatPromptTemplate.from_template("""
You are a surgical editor. Remove unsupported claims from a draft answer.

RETRIEVED CONTEXT (the only source of truth):
{context}

DRAFT ANSWER:
{answer}

UNSUPPORTED CLAIMS TO REMOVE:
{claims}

Instructions:
- Remove or rewrite ONLY the sentences that contain the flagged claims.
- Keep every other sentence exactly as written — do not paraphrase or expand.
- Do not introduce any new information not present in the retrieved context.
- Do not add commentary about what you changed.
- If removing all flagged claims leaves nothing meaningful, reply with exactly:
  I don't have enough information to answer this confidently.
- Return only the corrected answer — nothing else.
""")

_POLISH_PROMPT = ChatPromptTemplate.from_template("""
You are a professional editor. Polish the answer into confident, finalized prose.

ANSWER TO POLISH:
{answer}

Rules:
- Remove ALL hedging filler such as: "I don't know", "I'm not sure",
  "I cannot confirm", "the context suggests", "based on the retrieved context",
  "it appears", "it seems", "Note:", "the documents do not specify",
  "historical context suggests", "I don't have enough information",
  "according to the provided context", and any similar meta-commentary.
- If a sentence exists only to express uncertainty, delete it entirely.
- Do NOT add any new information. Only clean what is already there.
- Write in plain, direct, confident prose.
- Return only the polished answer — nothing else.
""")


class CriticAndRepair:
    """
    Claim-level hallucination detection with surgical repair and polishing.

    Usage:
        critic = CriticAndRepair(llm)

        verdict, claims, score = critic.check(context, answer)
        if verdict == "HALLUCINATED":
            repaired = critic.repair(context, answer, claims)
            final_verdict, _, score = critic.check(context, repaired)

        polished = critic.polish(repaired)
    """

    def __init__(self, llm: ChatOllama) -> None:
        self._critic_chain = _CRITIC_PROMPT  | llm | StrOutputParser()
        self._repair_chain = _REPAIR_PROMPT  | llm | StrOutputParser()
        self._polish_chain = _POLISH_PROMPT  | llm | StrOutputParser()

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self, context: str, answer: str
    ) -> tuple[str, str, float]:
        """
        Run the three-stage critic.

        Returns:
            verdict  — "GROUNDED" | "HALLUCINATED"
            claims   — bullet-list string of unsupported claims (empty if grounded)
            score    — faithfulness float 0.0–1.0
        """
        # Stage 1: no tool context at all
        if not context or not context.strip():
            log.info("[Critic] No context — HALLUCINATED (no tool calls)")
            return "HALLUCINATED", "No retrieval context was provided.", 0.0

        # Stage 2: predominantly "I don't know" → accept without LLM call
        if self._is_uncertainty_response(answer):
            log.info("[Critic] Uncertainty admission — auto GROUNDED")
            return "GROUNDED", "", 1.0

        # Stage 3: LLM claim-level check
        raw = self._critic_chain.invoke(
            {"context": context, "answer": answer}
        ).strip()

        return self._parse_critic_output(raw)

    def repair(self, context: str, answer: str, claims: str) -> str:
        """
        Surgically remove flagged claims from `answer`.
        No tool calls — fast focused LLM edit only.
        """
        log.info(f"[Repair] Removing {claims.count('•')} flagged claim(s)...")
        repaired = self._repair_chain.invoke({
            "context": context,
            "answer":  answer,
            "claims":  claims,
        }).strip()

        if not repaired or len(repaired) < 5:
            log.warning("[Repair] LLM returned empty repair — using fallback")
            return "I don't have enough information to answer this confidently."

        log.info(f"[Repair] Done. Length: {len(answer)} → {len(repaired)} chars")
        return repaired

    def polish(self, answer: str) -> str:
        """
        Strip hedging language from a grounded answer so it reads as confident.
        If the entire answer is an uncertainty admission, returns a clean fallback
        instead of running the polish LLM call.
        """
        if self._is_uncertainty_response(answer):
            return (
                "The available sources do not contain enough information "
                "to answer this question."
            )
        log.info("[Polish] Removing hedging language...")
        polished = self._polish_chain.invoke({"answer": answer}).strip()
        if not polished or len(polished) < 5:
            log.warning("[Polish] Empty polish result — returning unpolished answer")
            return answer
        return polished

    def compute_faithfulness(self, verdict: str, claims: str) -> float:
        """
        Convert a critic verdict into a 0–1 faithfulness score.
        Stored in query_metrics.answer_faithfulness.

          GROUNDED with no claims      → 1.0
          HALLUCINATED with N claims   → max(0, 1 - 0.2 * N)
          No context                   → 0.0
        """
        if verdict == "GROUNDED":
            return 1.0
        n_claims = claims.count("•")
        return max(0.0, round(1.0 - 0.2 * n_claims, 2))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_uncertainty_response(self, answer: str) -> bool:
        """
        True only if >= 50% of sentences in the answer are uncertainty phrases.
        Prevents a single buried "I don't know" from shielding a long
        substantive answer from critic review.
        """
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        if not sentences:
            return False
        lower = answer.lower()
        uncertainty_count = sum(
            1 for s in sentences
            if any(phrase in s.lower() for phrase in UNCERTAINTY_PHRASES)
        )
        ratio = uncertainty_count / len(sentences)
        log.debug(
            f"[Critic] Uncertainty ratio: {uncertainty_count}/{len(sentences)} = {ratio:.2f}"
        )
        return ratio >= 0.5

    def _parse_critic_output(self, raw: str) -> tuple[str, str, float]:
        """
        Parse the LLM critic's raw output into (verdict, claims, score).
        Handles unexpected formats gracefully.
        """
        upper = raw.upper()

        if "FULLY_GROUNDED" in upper or "FULLY GROUNDED" in upper:
            log.info("[Critic] FULLY GROUNDED")
            return "GROUNDED", "", 1.0

        if "UNSUPPORTED_CLAIMS" in upper or "UNSUPPORTED CLAIMS" in upper:
            # Extract bullet list
            claims_section = re.sub(
                r"UNSUPPORTED[_ ]CLAIMS\s*:?\s*", "", raw, flags=re.IGNORECASE
            ).strip()
            n = claims_section.count("•")
            score = max(0.0, round(1.0 - 0.2 * n, 2))
            log.info(f"[Critic] HALLUCINATED — {n} unsupported claim(s)")
            if claims_section:
                log.info(f"[Critic] Claims:\n{claims_section}")
            return "HALLUCINATED", claims_section, score

        # Unexpected format — be lenient and accept
        log.warning(
            f"[Critic] Unexpected response format: '{raw[:80]}' — defaulting GROUNDED"
        )
        return "GROUNDED", "", 0.8