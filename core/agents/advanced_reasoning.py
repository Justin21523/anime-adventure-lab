# core/agents/advanced_reasoning.py
"""
Advanced reasoning agent with reflection and quality checks.

Builds on SimpleReasoningAgent to provide stronger planning and synthesis:
- Plan refinement using a structured JSON planning-review prompt.
- Result synthesis quality review using a structured JSON review prompt.
- Robust parsing that works with both JSON and simple bullet/numbered lists.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from .base_agent import AgentMemory, SimpleReasoningAgent

logger = logging.getLogger(__name__)


# --- Prompt templates -------------------------------------------------------


PLAN_REFINEMENT_PROMPT = """You are an expert planning reviewer for an autonomous tool-using AI agent.

You receive:
- a natural language task description
- a draft plan as a JSON array of step descriptions

Your job:
- Improve the draft plan so it is more robust and directly executable by other components.
- Remove redundant steps and add missing but important steps when necessary.
- Keep the plan reasonably short and high-signal.
- Assume the plan will be executed automatically with no further human clarification.

Task:
{task}

Draft plan (JSON array of strings):
{draft_plan}

Output format (JSON only, no extra commentary or markdown):

{{
  "steps": [
    {{
      "description": "Improved step description 1, concrete and actionable."
    }},
    {{
      "description": "Improved step description 2, if needed."
    }}
  ]
}}
"""


QUALITY_REVIEW_PROMPT = """You are a quality-review module for an autonomous AI agent.

You receive:
- the original task
- a linearized history of steps and tool calls (as JSON)
- the current summary of results produced by another agent component

Your job:
- Check the summary for missing key details, factual inconsistencies, or unclear explanations.
- Propose specific improvements when necessary.
- Optionally provide a revised summary that is clearer and more accurate.

Task:
{task}

Step history (JSON list of objects with at least 'type' and 'content'):
{history}

Current summary:
{current_summary}

Output format (JSON only, no extra commentary or markdown):

{{
  "overall_quality": "high | medium | low",
  "issues": [
    "Issue 1 (if any)",
    "Issue 2 (if any)"
  ],
  "improvements": [
    "Short suggestion 1",
    "Short suggestion 2"
  ],
  "revised_summary": "If the summary should be updated, provide a full improved version here. Otherwise, repeat the current summary."
}}
"""


class AdvancedReasoningAgent(SimpleReasoningAgent):
    """
    Enhanced agent that adds plan refinement and result review on top of
    SimpleReasoningAgent.

    Features:
    - Optional reflective planning: refines the base plan using PLAN_REFINEMENT_PROMPT.
    - Optional quality checks: runs a structured review of the final summary.
    """

    def __init__(
        self,
        reflection_enabled: bool = True,
        quality_checks: bool = True,
        reflection_max_tokens: int = 256,
        review_max_tokens: int = 256,
        max_merged_steps: int = 8,
        **kwargs: Any,
    ):
        """
        :param reflection_enabled: enable/disable plan refinement step.
        :param quality_checks: enable/disable synthesis quality review.
        :param reflection_max_tokens: max tokens for plan refinement LLM call.
        :param review_max_tokens: max tokens for quality review LLM call.
        :param max_merged_steps: upper bound for merged refined plan length.
        :param kwargs: forwarded to SimpleReasoningAgent.
        """
        kwargs.setdefault("name", "advanced_reasoning_agent")
        kwargs.setdefault(
            "description",
            "Advanced reasoning agent with reflective planning and verification.",
        )
        super().__init__(**kwargs)

        self.reflection_enabled = reflection_enabled
        self.quality_checks = quality_checks
        self.reflection_max_tokens = reflection_max_tokens
        self.review_max_tokens = review_max_tokens
        self.max_merged_steps = max_merged_steps

    # --------------------------------------------------------------------- #
    # Planning
    # --------------------------------------------------------------------- #

    async def plan_task(self, task_description: str) -> List[str]:
        """
        Plan task with optional reflective refinement.

        Flow:
        1. Call the base SimpleReasoningAgent.plan_task to get a draft plan.
        2. If reflection is enabled and an LLM adapter is available:
           - Call the LLM with PLAN_REFINEMENT_PROMPT.
           - Parse the returned JSON / text into a list of steps.
           - Merge with the original plan while deduplicating.
        """
        base_plan = await super().plan_task(task_description)
        if not base_plan:
            return base_plan

        # If reflection is disabled or no LLM adapter is available, keep the base plan.
        if not self.reflection_enabled or not getattr(self, "llm_adapter", None):
            return base_plan

        try:
            draft_plan_json = json.dumps(base_plan, ensure_ascii=False)
            prompt = PLAN_REFINEMENT_PROMPT.format(
                task=task_description, draft_plan=draft_plan_json
            )

            critique = await self._llm_text(
                prompt,
                max_tokens=self.reflection_max_tokens,
                temperature=0.2,
            )
            refined_steps = self._parse_steps(critique)

            if refined_steps:
                merged = self._merge_steps(base_plan, refined_steps)
                logger.debug(
                    "Advanced plan refinement complete: base=%d refined=%d merged=%d",
                    len(base_plan),
                    len(refined_steps),
                    len(merged),
                )
                return merged

            logger.debug("Advanced plan refinement produced no usable steps; using base plan.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Plan refinement skipped due to error: %s", exc)

        return base_plan

    # --------------------------------------------------------------------- #
    # Synthesis / quality review
    # --------------------------------------------------------------------- #

    async def synthesize_result(self, memory: AgentMemory) -> str:
        """
        Summarize results and optionally run a structured quality review.

        Flow:
        1. Call the base SimpleReasoningAgent.synthesize_result for an initial summary.
        2. If quality checks are enabled and an LLM adapter is available:
           - Call the LLM with QUALITY_REVIEW_PROMPT.
           - Parse the JSON review.
           - Optionally replace the summary with the revised version and append
             a human-readable review section.
        """
        base_summary = await super().synthesize_result(memory)

        if not self.quality_checks or not getattr(self, "llm_adapter", None):
            return base_summary

        try:
            # Normalize history into a simple JSON-serializable structure.
            history_payload: List[Dict[str, Any]] = []
            for step in getattr(memory, "step_history", []):
                # step is expected to be a dict-like object
                try:
                    history_payload.append(
                        {
                            "type": step.get("type"),
                            "content": step.get("content"),
                            "tool_name": step.get("tool_name"),
                            "metadata": step.get("metadata"),
                        }
                    )
                except Exception:  # noqa: BLE001
                    # Be defensive: don't fail the whole review because of one bad entry.
                    history_payload.append({"raw": str(step)})

            history_json = json.dumps(history_payload, ensure_ascii=False, default=str)

            prompt = QUALITY_REVIEW_PROMPT.format(
                task=getattr(memory, "task_description", ""),
                history=history_json,
                current_summary=base_summary,
            )

            review_raw = await self._llm_text(
                prompt,
                max_tokens=self.review_max_tokens,
                temperature=0.3,
            )
            review = self._parse_quality_review(review_raw)
            if not review:
                logger.debug(
                    "Advanced quality review returned no parsable JSON; "
                    "falling back to base summary."
                )
                return base_summary

            revised_summary = (
                str(review.get("revised_summary")).strip() or base_summary
            )
            overall_quality = review.get("overall_quality")
            issues = [str(i).strip() for i in (review.get("issues") or []) if str(i).strip()]
            improvements = [
                str(i).strip() for i in (review.get("improvements") or []) if str(i).strip()
            ]

            review_sections: List[str] = []
            if overall_quality:
                review_sections.append(f"Overall quality: {overall_quality}.")
            if issues:
                review_sections.append("Potential issues:\n- " + "\n- ".join(issues))
            if improvements:
                review_sections.append(
                    "Suggested improvements:\n- " + "\n- ".join(improvements)
                )

            if not review_sections:
                # If there is no additional content, just return the (possibly) revised summary.
                return revised_summary

            return (
                revised_summary.rstrip()
                + "\n\nQuality review:\n"
                + "\n".join(review_sections)
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Quality review skipped due to error: %s", exc)
            return base_summary

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _safe_json_loads(self, text: str) -> Any:
        """
        Try to parse JSON from a string. If direct parsing fails, attempt to
        extract the first JSON object/array substring.

        Raises json.JSONDecodeError if everything fails.
        """
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try object
            obj_start = text.find("{")
            obj_end = text.rfind("}")
            if 0 <= obj_start < obj_end:
                try:
                    return json.loads(text[obj_start : obj_end + 1])
                except json.JSONDecodeError:
                    pass

            # Try array
            arr_start = text.find("[")
            arr_end = text.rfind("]")
            if 0 <= arr_start < arr_end:
                return json.loads(text[arr_start : arr_end + 1])

            # Reraise original error
            raise

    def _parse_steps(self, text: Optional[str]) -> List[str]:
        """
        Parse steps from LLM text.

        Supports:
        - JSON object with a "steps" array, where each element is either a string
          or an object with a "description" field.
        - JSON array of strings or objects.
        - Fallback: plain text with bullet or numbered lists.
        """
        if not text:
            return []

        raw = str(text).strip()
        steps: List[str] = []

        # Try JSON first.
        try:
            data = self._safe_json_loads(raw)
            raw_steps: Optional[Any] = None

            if isinstance(data, dict) and "steps" in data:
                raw_steps = data["steps"]
            elif isinstance(data, list):
                raw_steps = data

            if raw_steps is not None:
                for item in raw_steps:
                    if isinstance(item, str):
                        candidate = item.strip()
                    elif isinstance(item, dict):
                        candidate = str(item.get("description", "")).strip()
                    else:
                        candidate = str(item).strip()

                    if candidate:
                        steps.append(candidate)

                if steps:
                    return steps
        except Exception:
            # If JSON parsing fails, we fall back to line-based parsing below.
            logger.debug("Failed to parse refined plan as JSON; falling back to text parsing.")

        # Fallback: parse bullet / numbered lines.
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            # Remove typical bullets or numbering like "1." / "1)" / "- "
            if line[:2].isdigit() and line[1:2] in {".", ")"}:
                line = line[2:].strip()
            elif line.startswith("- "):
                line = line[2:].strip()
            elif line.startswith("-"):
                line = line[1:].strip()

            if line:
                steps.append(line)

        return steps

    def _merge_steps(self, primary: List[str], secondary: List[str]) -> List[str]:
        """
        Merge two step lists while keeping order and uniqueness.

        - All steps from `primary` are kept in their original order.
        - New, non-duplicate steps from `secondary` are appended in order.
        - The result is truncated to self.max_merged_steps to prevent runaway plans.
        """
        seen = set()
        merged: List[str] = []

        for step in primary + secondary:
            key = step.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(step.strip())

        if self.max_merged_steps is not None:
            return merged[: self.max_merged_steps]
        return merged

    def _parse_quality_review(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Parse the quality review JSON returned by the LLM.

        Returns an empty dict if parsing fails.
        """
        if not text:
            return {}

        try:
            data = self._safe_json_loads(str(text))
            if isinstance(data, dict):
                return data
            logger.debug("Quality review JSON is not a dict: %r", data)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to parse quality-review JSON: %s; raw=%r", exc, text)

        return {}
