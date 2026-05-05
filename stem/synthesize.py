"""
stem/synthesize.py

Second phase of the stem agent transformation pipeline.

The synthesize phase transforms raw domain observations into a concrete,
structured agent configuration. It is the decision-making core of the stem:
it determines what the specialized agent will be, what it accepts, what it
rejects, and how it reasons.

Design decisions:
- Uses Pydantic structured outputs: guarantees a valid, typed AgentConfig
  on every run. Eliminates fragile json.loads parsing and silent fallbacks.
- Validates semantic quality via field_validators: structure alone is not
  enough — a system_prompt of 10 chars is structurally valid but useless.
- Retries once on quality failure: if validation fails, a single retry is
  attempted before raising. More than one retry suggests a prompt or domain
  problem, not randomness.
- Uses EAFP for temperature: attempts temperature=0 and falls back
  gracefully if the model does not support it.
- Accepts ObserveResult for typing: makes the inter-phase contract explicit
  and enables autocompletion and static analysis.
- Prompts the model to organize observations before synthesizing: reduces
  concept mixing and improves output consistency.
"""

import os
import time
from typing import TypedDict

import openai
from dotenv import load_dotenv
from openai import OpenAI

from core.logger import get_logger
from stem.observe import ObserveResult
from stem.schemas import TOOL_NAMES, AgentConfig

load_dotenv()

log = get_logger(__name__)
client = OpenAI(max_retries=3)

_model = os.getenv("OPENAI_MODEL_DEEP")
if not _model:
    raise ValueError("OPENAI_MODEL_DEEP not set in .env")

# How many times to retry if the model produces a semantically invalid config.
# More than one retry suggests a prompt or domain problem, not randomness.
_MAX_QUALITY_RETRIES = 1

_TOOL_DESCRIPTIONS = "\n".join(f"- {name}" for name in TOOL_NAMES)

_SYNTHESIZE_PROMPT = """
You are the synthesize phase of a stem agent.
You have received raw observations about how experts work in a domain.

Before synthesizing, organize the observations internally into:
- Steps experts follow
- Tools they use
- Quality signals
- Common pitfalls

Then synthesize from that structure into a concrete agent configuration.

Think carefully about:
- What kind of agent would excel in this domain?
- What specific tasks should it accept and reject?
- What expert knowledge should its system prompt encode?
- What tools does it genuinely need?
- How does it know when it has a good enough answer?

The system_prompt you generate will be used directly as the system prompt
for the specialized agent — it must encode real expert knowledge, not generic
advice. Be specific, thorough, and actionable.

Available tools in this system:
{_TOOL_DESCRIPTIONS}

Only request tools from this list.
"""


class SynthesizeResult(TypedDict):
    """Output contract for the synthesize phase."""

    config: AgentConfig  # Validated structured agent configuration
    model: str  # Model used for synthesis
    elapsed: float  # Time taken in seconds


def _call_model(domain: str, observations_content: str) -> AgentConfig:
    """
    Call the LLM to synthesize an agent configuration from observations.

    Uses Pydantic structured outputs to guarantee a valid AgentConfig.
    Retries once if the model produces a semantically invalid config.
    Uses EAFP for temperature handling.

    Args:
        domain: The problem domain being synthesized.
        observations_content: Raw text from the observe phase.

    Returns:
        A validated AgentConfig instance.

    Raises:
        ValueError: If the model returns an empty response or fails
                    quality validation after retries.
        openai.OpenAIError: If the API call fails.
    """
    messages = [
        {"role": "system", "content": _SYNTHESIZE_PROMPT},
        {
            "role": "user",
            "content": f"Domain: {domain}\n\nObservations:\n{observations_content}",
        },
    ]

    def _parse(response) -> AgentConfig:
        if not response.choices or not response.choices[0].message.parsed:
            raise ValueError("Model returned an empty or unparseable response.")
        return response.choices[0].message.parsed

    def _create(**kwargs) -> AgentConfig:
        try:
            response = client.beta.chat.completions.parse(
                model=_model,
                temperature=0,
                messages=messages,
                response_format=AgentConfig,
                **kwargs,
            )
        except openai.BadRequestError as e:
            if "temperature" in str(e):
                log.debug(
                    "Model '%s' does not support temperature — retrying with default.",
                    _model,
                )
                response = client.beta.chat.completions.parse(
                    model=_model,
                    messages=messages,
                    response_format=AgentConfig,
                    **kwargs,
                )
            else:
                raise
        return _parse(response)

    for attempt in range(1, _MAX_QUALITY_RETRIES + 2):
        try:
            return _create()
        except Exception as e:
            if attempt <= _MAX_QUALITY_RETRIES:
                log.warning(
                    "Quality validation failed (attempt %d/%d): %s — retrying.",
                    attempt,
                    _MAX_QUALITY_RETRIES + 1,
                    e,
                )
            else:
                raise


def synthesize(domain: str, observations: ObserveResult) -> SynthesizeResult:
    """
    Synthesize a specialized agent configuration from domain observations.

    Transforms the raw observations gathered by the observe phase into a
    concrete, validated AgentConfig that the generate phase can consume.

    Args:
        domain: The name of the problem domain.
        observations: ObserveResult from the observe phase.

    Returns:
        SynthesizeResult with validated AgentConfig, model, and timing.

    Raises:
        ValueError: If the model returns an empty or semantically invalid
                    response after retries.
        openai.OpenAIError: If the API call fails.
    """
    log.info("Synthesizing agent configuration for domain '%s'...", domain)

    start = time.time()
    config = _call_model(domain, observations["content"])
    elapsed = time.time() - start

    log.info(
        "Done in %.2fs. Agent: '%s'",
        elapsed,
        config.identity,
    )

    return SynthesizeResult(
        config=config,
        model=_model,
        elapsed=elapsed,
    )


def _main() -> None:
    from stem.observe import observe

    observations = observe("Technical Research")
    result = synthesize("Technical Research", observations)
    print(result["config"].model_dump_json(indent=2))


if __name__ == "__main__":
    _main()
