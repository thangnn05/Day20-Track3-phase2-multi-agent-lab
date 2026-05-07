"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# Rough pricing per 1M tokens (gpt-4o-mini defaults)
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = _PRICING.get(model, (0.15, 0.60))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


class LLMClient:
    """Provider-agnostic LLM client using OpenAI SDK."""

    def __init__(self, model: str | None = None, temperature: float = 0.2) -> None:
        settings = get_settings()
        self._model = model or settings.openai_model
        self._temperature = temperature
        self._client = OpenAI(api_key=settings.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry, token logging, and cost estimation."""

        logger.info("LLM call: model=%s, system_len=%d, user_len=%d",
                     self._model, len(system_prompt), len(user_prompt))

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        choice = response.choices[0]
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = _estimate_cost(self._model, input_tokens, output_tokens)

        logger.info("LLM response: tokens_in=%s, tokens_out=%s, cost=$%s",
                     input_tokens, output_tokens, f"{cost:.6f}" if cost else "N/A")

        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
