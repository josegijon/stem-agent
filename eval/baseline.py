"""
eval/baseline.py

Minimal hardcoded baseline agent for evaluation.

This agent uses a generic system prompt with no domain specialization.
It has access to the same tools as the stem-generated agent to ensure
a fair comparison — the only variable is the quality of the system prompt
and the absence of domain-aware task classification.

Design decisions:
- Generic system prompt: deliberately minimal to maximize the contrast
  with the stem-generated agent. The baseline represents what you get
  without the stem transformation.
- No domain classifier: the baseline accepts all tasks. This is the
  natural behavior of a non-specialized agent.
- Same tools as stem agent: ensures the comparison isolates prompt
  quality, not tool availability.
"""

from agent.tools import AGENT_TOOLS

BASELINE_CONFIG = {
    "domain": "general",
    "identity": "Generic assistant with no domain specialization",
    "system_prompt": "You are a helpful assistant. Answer questions and complete tasks to the best of your ability.",
    "tools": list(AGENT_TOOLS.keys()) + ["direct_answer"],
    "accepts": [],
    "rejects": [],
    "stop_criteria": "Stop when you have answered the question.",
    "quality_criteria": "A good answer is accurate and complete.",
    "generated_by": "hardcoded-baseline",
    "stem_model": "none",
    "elapsed_stem": 0.0,
}
