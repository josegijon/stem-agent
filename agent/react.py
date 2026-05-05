"""
agent/react.py

ReAct (Reason + Act) loop for the specialized agent.

The ReAct pattern interleaves reasoning and action in a loop:
    Thought → the agent reasons about what to do next
    Action  → the agent calls a tool
    Observation → the agent reads the result and continues

The loop runs until the agent calls direct_answer (final answer reached)
or MAX_STEPS is exceeded (safety limit).

Design decisions:
- Uses OPENAI_MODEL_FAST: the ReAct loop runs on every task and may
  execute multiple steps. A fast, cheap model keeps latency acceptable.
- Domain checking uses the LLM: accepts/rejects are natural language
  descriptions. Only a LLM can reason about whether a task fits — regex
  or keyword matching would produce false positives and negatives.
- Domain checking fails open on API error: if the classifier fails,
  the agent attempts the task rather than silently rejecting it.
- MAX_STEPS from environment: configurable without code changes.
  Defaults to 8 — enough for complex research tasks, bounded for safety.
- _parse_action returns None only if no Action line is found: the agent
  may produce a final prose response without following ReAct format.
  We return it as-is rather than discarding it.
- accepts/rejects are not truncated in the system prompt: truncating
  domain boundaries risks the agent accepting tasks it should reject.
"""

import os
import re
from typing import TypedDict

import openai
from dotenv import load_dotenv
from openai import OpenAI

from agent.tools import TOOLS, execute_tool
from core.logger import get_logger

load_dotenv()

log = get_logger(__name__)
client = OpenAI(max_retries=3)

_fast_model = os.getenv("OPENAI_MODEL_FAST")
if not _fast_model:
    raise ValueError("OPENAI_MODEL_FAST not set in .env")

_MAX_STEPS: int = int(os.getenv("REACT_MAX_STEPS", "8"))

_REACT_INSTRUCTIONS = """
You reason step by step using this exact format:

Thought: reason about what you need to do next
Action: tool_name
Action Input: the input for the tool

Available tools: {tools}

After receiving an Observation, continue with another Thought/Action/Action Input.
When you have enough information to answer completely, use:

Action: direct_answer
Action Input: your complete, detailed answer here

Rules:
- Always start with a Thought
- Action must be exactly one of the available tools
- Never skip the Thought step
- If a tool returns an error, reason about it and try a different approach
- Do not repeat the same Action Input twice
"""

_CLASSIFIER_PROMPT = """
You are a binary classification router.
Your job is to decide if a given task should be routed to a specialized agent.

Agent Domain: {domain}
Agent Identity: {identity}
Available Tools: {tools}

Typical accepted tasks:
{accepts}

Explicitly rejected tasks:
{rejects}

Rules:
1. If the task is a question or request that can be answered using the agent's domain knowledge OR its available tools, output ACCEPT.
2. The task does not need to perfectly match the 'accepted tasks' list. Broad domain alignment is enough.
3. ONLY output REJECT if the task clearly falls into the 'rejected tasks' list or belongs to a completely different profession (e.g., asking a researcher to write unit tests or cook a recipe).
4. When in doubt, lean towards ACCEPT.
5. The available tools listed above are explicitly permitted. Any task that can be completed using only those tools should be ACCEPTED.

Output EXACTLY one word: ACCEPT or REJECT. No other text.
"""


class RunResult(TypedDict):
    """Output contract for a single agent run."""

    answer: str  # Final answer or rejection message
    steps: int  # Number of ReAct steps executed
    accepted: bool  # Whether the task was accepted


def _build_system_prompt(agent_config: dict) -> str:
    """
    Build the full system prompt for the ReAct loop.

    Combines the agent's expert system prompt with domain boundaries
    and ReAct format instructions.

    Args:
        agent_config: Executable agent configuration dict.

    Returns:
        Full system prompt string.
    """
    tools_list = ", ".join(agent_config.get("tools", list(TOOLS.keys())))

    accepts_str = "\n".join(f"- {a}" for a in agent_config.get("accepts", []))
    rejects_str = "\n".join(f"- {r}" for r in agent_config.get("rejects", []))

    react_instructions = _REACT_INSTRUCTIONS.format(tools=tools_list)

    return (
        f"{agent_config.get('system_prompt', 'You are a helpful agent.')}\n\n"
        f"---\n"
        f"Domain: {agent_config.get('domain', 'general')}\n\n"
        f"Tasks you accept:\n{accepts_str}\n\n"
        f"Tasks you reject:\n{rejects_str}\n\n"
        f"Stop criteria: {agent_config.get('stop_criteria', '')}\n\n"
        f"---\n"
        f"{react_instructions}"
    )


def _is_out_of_domain(task: str, agent_config: dict) -> bool:
    """
    Classify whether a task is outside the agent's domain.

    Uses the LLM to reason over natural language accept/reject criteria.
    Fails open on API error — the agent attempts the task rather than
    silently rejecting it.

    Args:
        task: The task string to classify.
        agent_config: Executable agent configuration dict.

    Returns:
        True if the task should be rejected, False otherwise.
    """
    rejects = agent_config.get("rejects", [])
    if not rejects:
        return False

    accepts_str = "\n".join(f"- {a}" for a in agent_config.get("accepts", []))
    rejects_str = "\n".join(f"- {r}" for r in rejects)

    try:
        response = client.chat.completions.create(
            model=_fast_model,
            messages=[
                {
                    "role": "system",
                    "content": _CLASSIFIER_PROMPT.format(
                        identity=agent_config.get("identity", ""),
                        domain=agent_config.get("domain", ""),
                        tools=", ".join(agent_config.get("tools", [])),
                        accepts=accepts_str,
                        rejects=rejects_str,
                    ),
                },
                {"role": "user", "content": f"Task: {task}"},
            ],
        )
        decision = response.choices[0].message.content.strip().upper()
        is_rejected = "REJECT" in decision and "ACCEPT" not in decision
        return is_rejected
    except openai.OpenAIError as e:
        log.warning(
            "Domain classifier failed — failing open and attempting task: %s", e
        )
        return False


def _parse_action(text: str) -> tuple[str, str] | None:
    """
    Parse an Action and Action Input from a ReAct step.

    Args:
        text: Raw assistant message text.

    Returns:
        Tuple of (action, action_input) if found, None otherwise.
        Returns None if the model produced a response without an Action
        line — the caller should treat this as a final answer.
    """
    action_match = re.search(r"Action:\s*(.+)", text)
    input_match = re.search(
        r"Action Input:\s*(.+?)(?=\nThought:|\nAction:|\Z)", text, re.DOTALL
    )

    if not action_match:
        return None

    action = action_match.group(1).strip()
    action_input = input_match.group(1).strip() if input_match else ""

    return action, action_input


def run(task: str, agent_config: dict) -> RunResult:
    """
    Run a task through the ReAct loop.

    Checks domain compatibility, then iterates through Thought/Action/
    Observation steps until the agent calls direct_answer or MAX_STEPS
    is reached.

    Args:
        task: The task string to execute.
        agent_config: Executable agent configuration dict.

    Returns:
        RunResult with the final answer, step count, and acceptance status.
    """
    domain = agent_config.get("domain", "unknown")
    log.info("Received task: %s", task)

    log.info("Checking domain compatibility...")
    if _is_out_of_domain(task, agent_config):
        accepts_preview = ", ".join(agent_config.get("accepts", [])[:3])
        rejection = (
            f"This task is outside my domain ({domain}). "
            f"I am specialized in: {accepts_preview}... "
            f"Please use an agent specialized for this type of task."
        )
        log.info("Task rejected.")
        return RunResult(answer=rejection, steps=0, accepted=False)

    log.info("Task accepted. Starting ReAct loop...")

    system_prompt = _build_system_prompt(agent_config)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    for step in range(1, _MAX_STEPS + 1):
        log.info("Step %d/%d", step, _MAX_STEPS)

        try:
            response = client.chat.completions.create(
                model=_fast_model,
                messages=messages,
            )
        except openai.OpenAIError as e:
            log.error("LLM call failed at step %d: %s", step, e)
            return RunResult(
                answer=f"Agent failed at step {step}: {e}",
                steps=step,
                accepted=True,
            )

        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})

        log.debug("Step %d response: %s", step, assistant_message[:200])

        parsed = _parse_action(assistant_message)
        if not parsed:
            # Model produced a response without following ReAct format.
            # Treat as final answer rather than discarding.
            log.info("No action parsed at step %d — treating as final answer.", step)
            return RunResult(answer=assistant_message, steps=step, accepted=True)

        action, action_input = parsed

        if action == "direct_answer":
            log.info("Final answer reached at step %d.", step)
            return RunResult(answer=action_input, steps=step, accepted=True)

        observation = execute_tool(action, action_input)
        messages.append({"role": "user", "content": f"Observation: {observation}"})
        log.debug("Observation: %s", observation[:150])

    log.warning("Max steps (%d) reached without a final answer.", _MAX_STEPS)
    return RunResult(
        answer="Max steps reached without a final answer.",
        steps=_MAX_STEPS,
        accepted=True,
    )


def _main() -> None:
    import json
    from pathlib import Path

    config_path = Path("config/generated/technical_research.json")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print("Please run 'python main.py' first to generate the domain config.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        agent_config = json.load(f)

    tests = [
        # --- IN-DOMAIN: should ACCEPT ---
        (
            "Test 1: Web search — technical comparison",
            "What are the main differences between PostgreSQL and SQLite for read-heavy workloads?",
        ),
        (
            "Test 2: Code execution — basic computation",
            "Write and execute a Python snippet to find the sha256 hash of the exact string 'JetBrains Internship'. Return the final hash.",
        ),
        (
            "Test 3: Multi-tool — search then compute",
            "Search for the time complexity of binary search. Then write and execute Python code that measures how many comparisons it takes to find a value in a sorted list of 1000 elements.",
        ),
        (
            "Test 4: Research question — state of the art",
            "What is the current state of the art in retrieval-augmented generation for question answering?",
        ),
        (
            "Test 5: Code execution — data analysis",
            "Write and execute Python code that generates 100 random numbers from a normal distribution and computes their mean and standard deviation.",
        ),
        (
            "Test 6: Technical trade-off analysis",
            "What are the trade-offs between transformer-based models and RNNs for sequence modeling tasks?",
        ),
        # --- OUT-OF-DOMAIN: should REJECT ---
        (
            "Test 7: Reject — unit testing",
            "Write unit tests for a Python function that sorts a list.",
        ),
        (
            "Test 8: Reject — cooking",
            "Give me a recipe for chocolate chip cookies.",
        ),
        (
            "Test 9: Reject — creative writing",
            "Write a short story about a robot who learns to paint.",
        ),
        (
            "Test 10: Reject — legal advice",
            "Should I incorporate my startup as an LLC or a C-Corp?",
        ),
        (
            "Test 11: Reject — UI development",
            "Build me a React component for a login form with email and password fields.",
        ),
    ]

    for title, task in tests:
        print(f"\n{'=' * 60}")
        print(f"=== {title} ===")
        print(f"{'=' * 60}")
        result = run(task, agent_config)
        print(f"\nFINAL ANSWER ({result['steps']} steps):\n{result['answer']}")
        print("\n" + "-" * 60)


if __name__ == "__main__":
    _main()
