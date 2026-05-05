"""
main.py

Entry point for the stem agent system.

Orchestrates two phases:
1. Stem phase: observes the domain, synthesizes a configuration, and
   generates a specialized agent. Skips transformation if a config
   already exists on disk.
2. Agent phase: runs the specialized agent in an interactive loop,
   accepting tasks from the user until 'exit' is entered.

Design decisions:
- Domain comes from STEM_DOMAIN env var: the domain is an operator
  decision, not a user decision. The user interacts with the specialized
  agent, not with the transformation process.
- main() contains no business logic: it delegates entirely to
  run_stem_phase and run_agent_phase. Each function has a single
  responsibility.
"""

import os

from dotenv import load_dotenv

from agent.react import run
from core.logger import get_logger
from stem.generate import generate, load_existing
from stem.observe import observe
from stem.synthesize import synthesize

load_dotenv()

log = get_logger(__name__)


def run_stem_phase(domain: str) -> dict:
    """
    Run the stem transformation phase for the given domain.

    Loads an existing agent config from disk if available, skipping
    the expensive transformation. Otherwise runs observe → synthesize
    → generate and returns the resulting config.

    Args:
        domain: The problem domain, e.g. 'Technical Research'.

    Returns:
        Executable agent configuration dict.
    """
    existing = load_existing(domain)
    if existing:
        log.info("Found existing config for '%s'. Skipping transformation.", domain)
        return existing

    log.info("No existing config found. Starting transformation for '%s'...", domain)

    observations = observe(domain)
    synthesis = synthesize(domain, observations)
    result = generate(domain, synthesis, observations)

    return result["config"]


def run_agent_phase(agent_config: dict) -> None:
    """
    Run the specialized agent in an interactive loop.

    Accepts tasks from the user until 'exit' is entered.
    Each task is routed through the ReAct loop and the result
    is printed to stdout.

    Args:
        agent_config: Executable agent configuration dict.
    """
    print(f"\n[AGENT] Specialized agent ready: {agent_config.get('identity', '')}\n")
    print("[AGENT] Type 'exit' to quit.\n")
    print("-" * 60)

    try:
        while True:
            task = input("\nTask: ").strip()

            if task.lower() == "exit":
                log.info("Shutting down.")
                break

            if not task:
                continue

            result = run(task, agent_config)

            print(f"\n{'=' * 60}")
            if not result["accepted"]:
                print(f"REJECTED:\n{result['answer']}")
            else:
                print(f"ANSWER ({result['steps']} steps):\n{result['answer']}")
            print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print()
        log.info("Interrupted. Shutting down.")


def main() -> None:
    """
    Main entry point for the stem agent system.

    Reads the domain from the STEM_DOMAIN environment variable,
    runs the stem phase to obtain or load a specialized agent config,
    then starts the interactive agent loop.
    """
    print("=" * 60)
    print("        STEM AGENT")
    print("=" * 60)
    print()

    domain = os.getenv("STEM_DOMAIN")
    if not domain:
        raise ValueError("STEM_DOMAIN not set in .env")

    log.info("Domain: %s", domain)

    agent_config = run_stem_phase(domain)
    run_agent_phase(agent_config)


if __name__ == "__main__":
    main()
