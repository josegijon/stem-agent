import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from agent.tools import execute_tool, TOOLS

load_dotenv()

client = OpenAI()
fast_model = os.getenv("OPENAI_MODEL_FAST", "gpt-4o")
MAX_STEPS = 8

REACT_INSTRUCTIONS = """
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


def _build_system_prompt(agent_config: dict) -> str:
    tools_list = ", ".join(agent_config.get("tools", list(TOOLS.keys())))

    base = agent_config.get(
        "system_prompt", "You are a helpful technical research agent."
    )
    domain = agent_config.get("domain", "general")
    accepts = agent_config.get("accepts", [])
    rejects = agent_config.get("rejects", [])
    stop_criteria = agent_config.get("stop_criteria", "")

    accepts_str = "\n".join(f"- {a}" for a in accepts[:5])
    rejects_str = "\n".join(f"- {r}" for r in rejects[:5])

    react_instructions = REACT_INSTRUCTIONS.format(tools=tools_list)

    return f"""{base}

---
Domain: {domain}
Tasks you accept:
{accepts_str}

Tasks you reject:
{rejects_str}

Stop criteria: {stop_criteria}

---
{react_instructions}
"""


def _is_out_of_domain(task: str, agent_config: dict) -> bool:
    rejects = agent_config.get("rejects", [])
    if not rejects:
        return False

    rejects_str = "\n".join(f"- {r}" for r in rejects)
    accepts_str = "\n".join(f"- {a}" for a in agent_config.get("accepts", []))

    response = client.chat.completions.create(
        model=fast_model,
        messages=[
            {
                "role": "system",
                "content": f"""You are a task classifier for a specialized agent.
The agent accepts these types of tasks:
{accepts_str}

The agent rejects these types of tasks:
{rejects_str}

Respond with only 'ACCEPT' or 'REJECT'.""",
            },
            {"role": "user", "content": f"Task: {task}"},
        ],
    )

    decision = response.choices[0].message.content.strip().upper()
    return decision == "REJECT"


def _parse_action(text: str) -> tuple[str, str] | None:
    action_match = re.search(r"Action:\s*(.+)", text)
    input_match = re.search(
        r"Action Input:\s*(.+?)(?=\nThought:|\nAction:|\Z)", text, re.DOTALL
    )

    if not action_match:
        return None

    action = action_match.group(1).strip()
    action_input = input_match.group(1).strip() if input_match else ""

    return action, action_input


def run(task: str, agent_config: dict) -> str:
    domain = agent_config.get("domain", "unknown")
    print(f"\n[AGENT/{domain}] Received task: {task}")

    print(f"[AGENT/{domain}] Checking domain compatibility...")
    if _is_out_of_domain(task, agent_config):
        rejection = (
            f"This task is outside my domain ({domain}). "
            f"I am specialized in: {', '.join(agent_config.get('accepts', [])[:3])}... "
            f"Please use an agent specialized for this type of task."
        )
        print(f"[AGENT/{domain}] Task rejected.\n")
        return rejection

    print(f"[AGENT/{domain}] Task accepted. Starting ReAct loop...\n")

    system_prompt = _build_system_prompt(agent_config)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"[AGENT/{domain}] Step {step}/{MAX_STEPS}")

        response = client.chat.completions.create(model=fast_model, messages=messages)

        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})

        print(f"  {assistant_message[:200]}...")

        parsed = _parse_action(assistant_message)
        if not parsed:
            return assistant_message

        action, action_input = parsed

        if action == "direct_answer":
            print(f"\n[AGENT/{domain}] Final answer reached at step {step}.\n")
            return action_input

        observation = execute_tool(action, action_input)
        observation_message = f"Observation: {observation}"
        messages.append({"role": "user", "content": observation_message})
        print(f"  Observation: {observation[:150]}...\n")

    return "Max steps reached without a final answer."


if __name__ == "__main__":
    import json
    from pathlib import Path

    config_path = Path("config/generated/technical_research.json")
    with open(config_path, "r", encoding="utf-8") as f:
        agent_config = json.load(f)

    print("=== Test 1: In-domain task ===")
    result = run(
        "What are the main differences between PostgreSQL and SQLite for read-heavy workloads?",
        agent_config,
    )
    print(f"\nFINAL ANSWER:\n{result}")

    print("\n=== Test 2: Out-of-domain task ===")
    result = run(
        "Write unit tests for a Python function that sorts a list", agent_config
    )
    print(f"\nFINAL ANSWER:\n{result}")
