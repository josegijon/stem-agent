"""
stem/observe.py

First phase of the stem agent transformation pipeline.

The observe phase gathers raw knowledge about how experts work in a given
domain. It makes no decisions — it only collects material for the synthesize phase to process.

Design decisions:
- Uses OPENAI_MODEL_DEEP: observation quality is the foundation of the entire pipeline. A weaker model produces shallower observations and a weaker agent.
- Uses temperature=0 where supported: observations must be reproducible across runs to allow meaningful comparison between experiments. Models that do not support custom temperature (e.g. gpt-5, o-series) fall back to their default via EAFP — the parameter is omitted on BadRequestError rather than maintaining a hardcoded list of incompatible models.
- Validates minimum length: protects against degenerate model outputs that would silently propagate bad data through the pipeline.
"""

import os
import time
from typing import TypedDict

from dotenv import load_dotenv
import openai
from openai import OpenAI

from core.logger import get_logger

load_dotenv()

log = get_logger(__name__)
client = OpenAI(max_retries=3)

# The deep model is used intentionally — observation quality determines
# the quality of the generated agent. Do not replace with a faster model.
_model = os.getenv("OPENAI_MODEL_DEEP")
if not _model:
    raise ValueError("OPENAI_MODEL_DEEP not set in .env")

# Minimum acceptable observation length. Outputs below this threshold
# indicate the model failed to research the domain properly.
_MIN_LENGTH = 2000

# This prompt is intentionally domain-agnostic — it must work for any domain
# without modification. Do not add domain-specific knowledge here.
_OBSERVE_PROMPT = """
You are the observe phase of a stem agent.
Your job is to research how experts actually work in a given domain.

Research and describe:
1. What steps do experts follow when working in this domain?
2. What tools do they use?
3. What makes a good result in this domain?
4. What are common mistakes or pitfalls?

Avoid vague statements.
Prefer concrete, actionable steps.
Use bullet points.
Do not generalize.
"""


class ObserveResult(TypedDict):
    """Output contract for the observe phase."""

    content: str  # Raw observations about the domain
    model: str  # Model used for this observation
    elapsed: float  # Time taken in seconds
    tokens_estimate: (
        int  # Approximate token count (chars // 4), useful for cost tracking
    )


def _call_model(domain: str) -> str:
    """
    Call the LLM to observe the given domain.

    Attempts to use temperature=0 for reproducibility. If the model does not
    support custom temperature (e.g. gpt-5, o-series), falls back to the
    model default automatically.

    Args:
        domain: The problem domain to research.

    Returns:
        Raw text content from the model.

    Raises:
        ValueError: If the model returns an empty response.
        openai.OpenAIError: If the API call fails for reasons other than
                            unsupported temperature.
    """
    messages = [
        {"role": "system", "content": _OBSERVE_PROMPT},
        {
            "role": "user",
            "content": f"Research how experts approach and execute tasks in this domain: {domain}",
        },
    ]

    try:
        response = client.chat.completions.create(
            model=_model,
            temperature=0,
            messages=messages,
        )
    except openai.BadRequestError as e:
        if "temperature" in str(e):
            log.debug(
                "Model '%s' does not support temperature — retrying with default.",
                _model,
            )
            response = client.chat.completions.create(
                model=_model,
                messages=messages,
            )
        else:
            raise

    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Model returned an empty response.")

    return response.choices[0].message.content


def _validate(content: str) -> None:
    """
    Validate that the observation meets minimum quality requirements.

    Args:
        content: The raw observation text to validate.

    Raises:
        ValueError: If the content is too short to be useful.
    """
    if len(content) < _MIN_LENGTH:
        raise ValueError(
            f"Observation too short ({len(content)} chars, minimum {_MIN_LENGTH}). "
            "The model may have failed to research the domain properly."
        )


def observe(domain: str) -> ObserveResult:
    """
    Observe how experts work in a given domain.

    Orchestrates the full observation phase: calls the model, validates
    the output, and returns a structured result for the synthesize phase.

    Args:
        domain: The name of the problem domain to observe.
                Should be a short label, e.g. 'Technical Research' or 'QA'.

    Returns:
        ObserveResult with raw content and metadata (model, timing, cost estimate).

    Raises:
        ValueError: If the model returns empty or insufficient content.
        openai.OpenAIError: If the API call fails.
    """
    log.info("Observing domain '%s' with model '%s'...", domain, _model)

    start = time.time()
    content = _call_model(domain)
    elapsed = time.time() - start

    _validate(content)

    log.info("Done in %.2fs. Gathered ~%d tokens.", elapsed, len(content) // 4)

    return ObserveResult(
        content=content,
        model=_model,
        elapsed=elapsed,
        tokens_estimate=len(content) // 4,
    )


def _main() -> None:
    result = observe("Technical Research")
    print(result["content"])


if __name__ == "__main__":
    _main()
