import os
from pathlib import Path
from dotenv import load_dotenv

from stem.observe import observe
from stem.synthesize import synthesize
from stem.generate import generate, load_existing
from agent.react import run

load_dotenv()

GENERATED_DIR = Path("config/generated")


def run_stem_phase(domain: str) -> dict:
    existing = load_existing(domain)
    if existing:
        print(
            f"[STEM] Found existing config for '{domain}'. Skipping transformation.\n"
        )
        return existing

    print(
        f"[STEM] No existing config found. Starting transformation for '{domain}'...\n"
    )

    observations = observe(domain)
    synthesis = synthesize(domain, observations)
    result = generate(domain, synthesis, observations)
    agent_config = result["config"]

    return agent_config


def run_agent_phase(agent_config: dict) -> None:

    print(f"[AGENT] Specialized agent ready: {agent_config.get('identity', '')}\n")
    print("[AGENT] Type 'exit' to quit.\n")
    print("-" * 60)

    while True:
        task = input("\nTask: ").strip()

        if task.lower() == "exit":
            print("\n[AGENT] Shutting down.")
            break

        if not task:
            continue

        result = run(task, agent_config)
        print(f"\n{'=' * 60}")
        print(f"ANSWER:\n{result}")
        print(f"{'=' * 60}")


def main():
    print("=" * 60)
    print("        STEM AGENT")
    print("=" * 60)
    print()

    domain = os.getenv("STEM_DOMAIN", "Technical Research")
    print(f"[STEM] Domain: {domain}\n")

    agent_config = run_stem_phase(domain)
    run_agent_phase(agent_config)


if __name__ == "__main__":
    main()
