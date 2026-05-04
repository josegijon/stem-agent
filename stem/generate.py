"""
stem/generate.py

Third and final phase of the stem agent transformation pipeline.

The generate phase takes the validated AgentConfig from synthesize and
produces a concrete, executable agent configuration. It is responsible for:

1. Mapping tools requested by the stem to tools that actually exist.
2. Persisting the configuration to disk for reuse across runs.
3. Saving a trace file with full provenance for reproducibility and eval.

Design decisions:
- Exact tool matching: the stem is informed of available tools in the
  synthesize prompt, so it requests tools by their exact names. Exact
  matching is simpler, faster, and more reliable than LLM-based semantic
  matching or keyword heuristics. Unknown tool names are discarded and
  logged as warnings.
- No fallback to all tools: an over-equipped agent is unpredictable and
  contaminates evaluation. If matching produces no tools, we fail explicitly.
- direct_answer is injected after matching: it is the ReAct loop stop
  signal, not a tool the stem should discover or request.
- Config persistence eliminates redundant stem runs: the first run is
  expensive (~4 min with GPT-5). Subsequent runs load from disk instantly.
- Trace includes git hash and timestamp: two traces for the same domain
  are distinguishable and fully reproducible without calling the API again.
"""

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from logger import get_logger
from stem.observe import ObserveResult
from stem.synthesize import SynthesizeResult
from stem.schemas import (
    AgentConfig,
    ExecutableAgentConfig,
)
from agent.tools import AGENT_TOOLS

load_dotenv()

log = get_logger(__name__)

GENERATED_DIR = Path("config/generated")

# Tools available in the system for the agent to use.
# direct_answer is always injected after matching — it is the ReAct stop
# signal, not a tool the stem should request.
_AVAILABLE_TOOLS: list[str] = list(AGENT_TOOLS.keys())
_ALWAYS_PRESENT: list[str] = ["direct_answer"]


class GenerateResult(TypedDict):
    """Output contract for the generate phase."""

    config: dict  # Serializable agent configuration ready for execution
    config_path: Path  # Path where the config was persisted
    trace_path: Path  # Path where the trace was persisted
    elapsed: float  # Time taken in seconds


def _slugify(domain: str) -> str:
    """
    Convert a domain name to a safe filesystem slug.

    Removes special characters, collapses whitespace and hyphens to
    underscores, and truncates to 50 characters to avoid path length
    issues on Windows and Unix.

    Args:
        domain: Raw domain name, e.g. 'Technical Research'.

    Returns:
        Safe slug, e.g. 'technical_research'.
    """
    slug = domain.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug[:50]


def _config_path(domain: str) -> Path:
    return GENERATED_DIR / f"{_slugify(domain)}.json"


def _trace_path(domain: str) -> Path:
    return GENERATED_DIR / f"{_slugify(domain)}_trace.json"


def _get_git_hash() -> str:
    """
    Return the current git commit hash for trace provenance.

    Returns 'unknown' if git is unavailable or the directory is not
    a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _match_tools(requested_tools: list[str]) -> tuple[list[str], list[str]]:
    """
    Match tools requested by the stem against tools that actually exist.

    The stem is informed of available tools in the synthesize prompt, so
    it requests tools by their exact names. Exact matching is simpler,
    faster, and more reliable than LLM-based or keyword-based approaches.

    Args:
        requested_tools: Tool names as requested by the stem agent.

    Returns:
        Tuple of (matched, discarded) tool name lists.
        matched: tools that exist in the system.
        discarded: tools requested by the stem that do not exist.
    """
    available = set(_AVAILABLE_TOOLS)
    seen: set[str] = set()
    matched: list[str] = []
    discarded: list[str] = []

    for tool in requested_tools:
        if tool in available and tool not in seen:
            matched.append(tool)
            seen.add(tool)
        elif tool not in available:
            discarded.append(tool)

    return matched, discarded


def _build_agent_config(
    config: AgentConfig,
    synthesis: SynthesizeResult,
    matched_tools: list[str],
) -> dict:
    """
    Build the serializable agent config dict from the Pydantic AgentConfig.

    Adds execution metadata not part of AgentConfig schema but useful
    for tracing and eval: generated_by, stem_model, elapsed_stem.

    Args:
        config: Validated AgentConfig from the synthesize phase.
        synthesis: Full SynthesizeResult for metadata.
        matched_tools: Final tool list including direct_answer.

    Returns:
        Serializable dict ready to be saved and consumed by the agent.
    """
    return {
        "domain": config.domain,
        "identity": config.identity,
        "system_prompt": config.system_prompt,
        "tools": matched_tools,
        "accepts": config.accepts,
        "rejects": config.rejects,
        "stop_criteria": config.stop_criteria,
        "quality_criteria": config.quality_criteria,
        "generated_by": "stem-agent",
        "stem_model": synthesis["model"],
        "elapsed_stem": synthesis["elapsed"],
    }


def _build_trace(
    domain: str,
    observations: ObserveResult,
    synthesis: SynthesizeResult,
    agent_config: dict,
    requested_tools: list[str],
    matched_tools: list[str],
    discarded_tools: list[str],
) -> dict:
    """
    Build the trace dict for reproducibility and evaluation.

    Captures full provenance of a generation run: inputs, outputs,
    timing, git hash, and tool matching quality.

    Args:
        domain: The problem domain.
        observations: ObserveResult from the observe phase.
        synthesis: SynthesizeResult from the synthesize phase.
        agent_config: Final serializable agent config.
        requested_tools: Tools originally requested by the stem.
        matched_tools: Tools matched to the available pool.
        discarded_tools: Tools requested but not available in the system.

    Returns:
        Trace dict ready to be serialized to disk.
    """
    return {
        "domain": domain,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _get_git_hash(),
        "observations": {
            "content": observations["content"],
            "model": observations["model"],
            "elapsed": observations["elapsed"],
            "tokens_estimate": observations["tokens_estimate"],
        },
        "synthesis": {
            "model": synthesis["model"],
            "elapsed": synthesis["elapsed"],
        },
        "tool_matching": {
            "requested": requested_tools,
            "matched": matched_tools,
            "discarded": discarded_tools,
            "injected": _ALWAYS_PRESENT,
        },
        "agent_config": agent_config,
    }


def _persist(path: Path, data: dict) -> None:
    """
    Persist a dict to a JSON file, creating parent directories as needed.

    Args:
        path: Target file path.
        data: Data to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_existing(domain: str) -> dict | None:
    """
    Load an existing agent config from disk if available.

    Args:
        domain: The problem domain to look up.

    Returns:
        The agent config dict if found, None otherwise.
    """
    path = _config_path(domain)
    if not path.exists():
        return None

    log.info("Found existing config for '%s'. Loading from disk.", domain)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate(
    domain: str,
    synthesis: SynthesizeResult,
    observations: ObserveResult,
) -> GenerateResult:
    """
    Generate a specialized agent configuration from a synthesis result.

    Orchestrates tool matching, config construction, validation,
    and persistence. Saves both the executable config and a full
    trace file for reproducibility and evaluation.

    Args:
        domain: The problem domain.
        synthesis: SynthesizeResult from the synthesize phase.
        observations: ObserveResult from the observe phase.

    Returns:
        GenerateResult with config dict, file paths, and timing.

    Raises:
        ValueError: If tool matching produces no usable tools.
    """
    log.info("Generating specialized agent for domain '%s'...", domain)
    start = time.time()

    config: AgentConfig = synthesis["config"]
    requested_tools = config.tools

    matched, discarded = _match_tools(requested_tools)

    if discarded:
        log.warning(
            "Stem requested %d tool(s) not available in this system: %s",
            len(discarded),
            discarded,
        )

    if not matched:
        raise ValueError(
            f"Tool matching produced no results for domain '{domain}'. "
            f"Requested: {requested_tools}. Available: {_AVAILABLE_TOOLS}. "
            "Ensure the synthesize prompt lists available tools correctly."
        )

    if len(matched) < len(_AVAILABLE_TOOLS):
        log.warning(
            "Only %d/%d available tools were matched. "
            "The agent may have reduced capabilities.",
            len(matched),
            len(_AVAILABLE_TOOLS),
        )

    final_tools = matched + _ALWAYS_PRESENT
    agent_config = _build_agent_config(config, synthesis, final_tools)

    trace = _build_trace(
        domain,
        observations,
        synthesis,
        agent_config,
        requested_tools,
        final_tools,
        discarded,
    )

    cfg_path = _config_path(domain)
    trc_path = _trace_path(domain)

    ExecutableAgentConfig(**agent_config)

    _persist(cfg_path, agent_config)
    _persist(trc_path, trace)

    elapsed = time.time() - start
    log.info(
        "Done in %.2fs. Config: %s | Trace: %s",
        elapsed,
        cfg_path,
        trc_path,
    )

    return GenerateResult(
        config=agent_config,
        config_path=cfg_path,
        trace_path=trc_path,
        elapsed=elapsed,
    )


def _main() -> None:
    from stem.observe import observe
    from stem.synthesize import synthesize

    domain = "Technical Research"

    existing = load_existing(domain)
    if existing:
        print(json.dumps(existing, indent=2))
    else:
        observations = observe(domain)
        synthesis_result = synthesize(domain, observations)
        result = generate(domain, synthesis_result, observations)
        print(json.dumps(result["config"], indent=2))


if __name__ == "__main__":
    _main()
