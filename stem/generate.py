import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

AVAILABLE_TOOLS = ["web_search", "run_code", "direct_answer"]
GENERATED_DIR = Path("config/generated")


def _match_tools(requested_tools: list) -> list:
    """
    Maps tools requested by the stem against tools that actually exist.
    Any requested tool that semantically relates to an available tool gets included.
    Unknown tools are discarded.
    """
    matched = set()

    tool_keywords = {
        "web_search": [
            "search",
            "scholar",
            "arxiv",
            "web",
            "browser",
            "internet",
            "google",
            "semantic",
            "research",
            "literature",
            "paper",
        ],
        "run_code": [
            "code",
            "python",
            "execute",
            "script",
            "compute",
            "notebook",
            "jupyter",
            "analysis",
            "pandas",
            "numpy",
            "scipy",
        ],
        "direct_answer": [
            "answer",
            "respond",
            "direct",
            "synthesize",
            "summarize",
            "write",
        ],
    }

    for requested in requested_tools:
        requested_lower = requested.lower()
        for available_tool, keywords in tool_keywords.items():
            if any(kw in requested_lower for kw in keywords):
                matched.add(available_tool)

    return list(matched) if matched else AVAILABLE_TOOLS


def _config_path(domain: str) -> Path:
    slug = domain.lower().replace(" ", "_")
    return GENERATED_DIR / f"{slug}.json"


def load_existing(domain: str) -> dict | None:
    path = _config_path(domain)
    if path.exists():
        print(
            f"[STEM/generate] Found existing config for '{domain}'. Loading from disk...\n"
        )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def generate(domain: str, synthesis: dict) -> dict:
    print(f"[STEM/generate] Generating specialized agent for '{domain}'...")

    start = time.time()
    config = synthesis["config"]

    matched_tools = _match_tools(config.get("tools", []))
    print(f"[STEM/generate] Tools requested by stem: {len(config.get('tools', []))}")
    print(f"[STEM/generate] Tools available in system: {matched_tools}")

    agent_config = {
        "domain": config.get("domain", domain),
        "identity": config.get("identity", ""),
        "system_prompt": config.get("system_prompt", ""),
        "tools": matched_tools,
        "accepts": config.get("accepts", []),
        "rejects": config.get("rejects", []),
        "stop_criteria": config.get("stop_criteria", ""),
        "quality_criteria": config.get("quality_criteria", ""),
        "generated_by": "stem-agent",
        "stem_model": synthesis.get("model", "unknown"),
        "elapsed_stem": synthesis.get("elapsed", 0),
    }

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = _config_path(domain)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(agent_config, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"[STEM/generate] Done in {elapsed:.2f}s.")
    print(f"[STEM/generate] Config saved to {path}\n")

    return agent_config


if __name__ == "__main__":
    from observe import observe
    from synthesize import synthesize

    domain = "Technical Research"

    existing = load_existing(domain)
    if existing:
        print(json.dumps(existing, indent=2))
    else:
        observations = observe(domain)
        synthesis = synthesize(domain, observations)
        agent_config = generate(domain, synthesis)
        print(json.dumps(agent_config, indent=2))
