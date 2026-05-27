"""LLM Critic: AI-driven scientific design reviewer.

This is the core AI contribution. The LLM Critic receives a structured prompt
(auto-generated from DomainSpec), examines the current design's performance,
and provides structured feedback with quantifiable targets that are mechanically
translated into differentiable loss terms via the Text-to-Math Bridge.

Uses LangChain + Ollama (local, free) by default. Supports OpenAI-compatible
APIs for ablation experiments comparing backends.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np

from ..domain_spec import DomainSpec
from ..text_to_math import RefinementFeedback


class LLMCritic:
    """LLM-powered scientific design critic.

    This is NOT a LangChain agent — it's a structured prompt → response loop.
    The LLM is used for its reasoning capability, not for tool use or action.
    """

    def __init__(self, domain: DomainSpec, model_name: str = "qwen2.5:14b",
                 temperature: float = 0.0, max_retries: int = 3,
                 backend: str = "ollama", base_url: str | None = None):
        """
        Args:
            domain: DomainSpec describing the design problem.
            model_name: LLM model identifier (e.g., "qwen2.5:14b", "gpt-4o-mini").
            temperature: LLM temperature (0.0 = deterministic).
            max_retries: Retries on JSON parse failure.
            backend: "ollama" or "openai_compatible".
            base_url: API base URL for OpenAI-compatible backends.
        """
        self.domain = domain
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.backend = backend
        self.base_url = base_url

        self._llm = None
        self._init_llm()

    def _init_llm(self):
        """Initialize the LLM backend."""
        if self.backend == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
                self._llm = ChatOllama(
                    model=self.model_name,
                    temperature=self.temperature,
                )
            except ImportError:
                raise ImportError(
                    "langchain-community is required for Ollama backend. "
                    "Install with: pip install langchain-community"
                )
        elif self.backend == "openai_compatible":
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError(
                    "langchain-openai is required for OpenAI-compatible backend. "
                    "Install with: pip install langchain-openai"
                )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def review(self, design_params: list[float], design_metrics: dict,
               constraint_value: float | None = None,
               use_minimal_prompt: bool = False) -> RefinementFeedback:
        """Review a design and return structured feedback.

        Args:
            design_params: Current design parameters (list of floats).
            design_metrics: Performance metrics dict.
            constraint_value: Constraint value (e.g., thickness in mm).
            use_minimal_prompt: If True, use minimal prompt without domain context
                (for A3 ablation: "Remove DomainSpec context").

        Returns:
            RefinementFeedback with verdict, weak_regions, etc.
        """
        # Build prompt
        if use_minimal_prompt:
            prompt = self.domain.to_minimal_prompt(design_params, design_metrics)
        else:
            prompt = self.domain.to_critic_prompt(
                design_params, design_metrics, constraint_value
            )

        # Call LLM with retry
        for attempt in range(self.max_retries):
            try:
                response = self._llm.invoke(prompt)
                text = response.content if hasattr(response, "content") else str(response)

                # Extract JSON from response
                data = self._extract_json(text)

                if data is None:
                    if attempt < self.max_retries - 1:
                        prompt = self._add_json_reminder(prompt, text)
                        continue
                    return self._fallback_feedback(
                        "Failed to parse JSON after retries", text
                    )

                feedback = RefinementFeedback.from_llm_json(data)
                return feedback

            except Exception as e:
                if attempt < self.max_retries - 1:
                    continue
                return self._fallback_feedback(f"LLM call failed: {e}")

        return self._fallback_feedback("Max retries exceeded")

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response text.

        Handles: ```json ... ``` blocks, inline JSON, and bare objects.
        """
        # Try ```json ... ``` block
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try ``` ... ``` block
        match = re.search(r"```\s*([\s\S]*?)\s*```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find a JSON object in the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _add_json_reminder(self, original_prompt: str, bad_response: str) -> str:
        """Add a reminder to output valid JSON when the LLM fails to do so."""
        reminder = (
            f"\n\nYOUR PREVIOUS RESPONSE DID NOT CONTAIN VALID JSON:\n"
            f"{bad_response[:200]}...\n\n"
            f"You MUST output a valid JSON object. "
            f"The JSON must be enclosed in ```json ... ``` code blocks. "
            f"Please try again with the exact JSON format specified above."
        )
        return original_prompt + reminder

    def _fallback_feedback(self, reason: str,
                           raw_text: str = "") -> RefinementFeedback:
        """Return conservative fallback feedback when LLM fails."""
        return RefinementFeedback(
            verdict="refine",
            weak_regions=[],
            physical_violations=[],
            causal_analysis=f"[FALLBACK] LLM Critic failed: {reason}. "
                            "Using default refinement.",
            exploration_suggestion="Continue multi-start gradient optimization.",
            confidence=0.0,
            raw_json={"error": reason, "raw_text": raw_text[:500]},
        )

    @property
    def token_count_last_call(self) -> int:
        """Estimated token count for the last call (if available)."""
        # Ollama doesn't expose token counts easily via LangChain
        # This is a rough estimate
        return -1


# ── Factory ──

def create_llm_critic(domain: DomainSpec, model: str = "qwen2.5:14b",
                      backend: str = "ollama", **kwargs) -> LLMCritic:
    """Factory function for LLM Critic.

    Args:
        domain: DomainSpec.
        model: Model name.
        backend: "ollama" or "openai_compatible".
        **kwargs: Passed to LLMCritic constructor.

    Returns:
        LLMCritic instance.
    """
    return LLMCritic(domain, model_name=model, backend=backend, **kwargs)
