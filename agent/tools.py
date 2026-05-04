"""
agent/tools.py

Tool implementations and registry for the specialized agent.

Each tool is a function that receives a string input and returns a string
output. The agent's ReAct loop calls tools by name via execute_tool().

Design decisions:
- web_search uses the LLM as a search proxy: the fast model synthesizes
  current information given a query. This is a known limitation — the model
  responds from training knowledge, not live web results. Documented here
  for eval and write-up purposes.
- run_code uses subprocess isolation: executing code in the same process
  as the agent risks crashing the entire system on errors. subprocess
  isolates execution with a hard timeout.
- direct_answer is the ReAct loop stop signal: it is not a real tool but
  is registered here so the loop can call it uniformly via execute_tool().
  It is excluded from AGENT_TOOLS (the pool available for matching) and
  injected explicitly by generate.py.
- AGENT_TOOLS is the single source of truth for available tools.
  generate.py and stem/schemas.py import from here to avoid drift.
"""

import os
import subprocess
import sys

import openai
from dotenv import load_dotenv
from openai import OpenAI

from logger import get_logger

load_dotenv()

log = get_logger(__name__)
client = OpenAI(max_retries=3)

_fast_model = os.getenv("OPENAI_MODEL_FAST")
if not _fast_model:
    raise ValueError("OPENAI_MODEL_FAST not set in .env")

_WEB_SEARCH_PROMPT = """
You are a web search assistant. The user gives you a search query.
Search for current, accurate information and return a concise summary.
Include specific facts, numbers, and sources where relevant.
Be direct and informative. Do not add commentary about the search process.
"""


def web_search(query: str) -> str:
    """
    Search for information given a query string.

    Uses the fast LLM as a search proxy. The model synthesizes information
    from its training knowledge — it does not perform live web retrieval.
    This is a known limitation: results may be outdated for recent events.

    Args:
        query: Natural language search query.

    Returns:
        Concise summary of relevant information, or an error message
        if the API call fails.
    """
    log.debug("web_search query: %s", query)

    try:
        response = client.chat.completions.create(
            model=_fast_model,
            messages=[
                {"role": "system", "content": _WEB_SEARCH_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        log.warning("web_search failed: %s", e)
        return f"Error: web_search failed — {e}"


def run_code(code: str) -> str:
    """
    Execute a Python code snippet and return stdout or stderr.

    Runs in an isolated subprocess to prevent crashes in the agent process.
    Hard timeout of 15 seconds to prevent infinite loops.

    Args:
        code: Python source code to execute.

    Returns:
        stdout if successful, stderr if the code raised an error,
        or a timeout/execution error message.
    """
    log.debug("run_code executing snippet (%d chars)", len(code))

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output else "Code executed successfully. No output."
        return f"Error:\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (15s limit)."
    except Exception as e:
        return f"Error: {e}"


def direct_answer(answer: str) -> str:
    """
    Pass through a final answer from the ReAct loop.

    This is the stop signal for the ReAct loop — not a real tool.
    It is registered in the tool registry so the loop can call it
    uniformly via execute_tool(), but it is excluded from AGENT_TOOLS
    so the stem never requests it as a capability.

    Args:
        answer: The final answer string from the agent.

    Returns:
        The answer unchanged.
    """
    return answer


# ─── Tool registry ────────────────────────────────────────────────────────────

# AGENT_TOOLS is the single source of truth for tools available to agents.
# Imported by stem/schemas.py to build ToolName and by generate.py for
# matching. direct_answer is excluded — it is a loop signal, not a capability.
AGENT_TOOLS: dict[str, callable] = {
    "web_search": web_search,
    "run_code": run_code,
}

# Full registry including direct_answer, used by the ReAct loop.
TOOLS: dict[str, callable] = {
    **AGENT_TOOLS,
    "direct_answer": direct_answer,
}


def execute_tool(name: str, tool_input: str) -> str:
    """
    Execute a tool by name.

    Args:
        name: Tool name. Must be a key in TOOLS.
        tool_input: Input string for the tool.

    Returns:
        Tool output string, or an error message if the tool does not exist.
    """
    if name not in TOOLS:
        return (
            f"Error: tool '{name}' does not exist. "
            f"Available tools: {list(TOOLS.keys())}"
        )
    return TOOLS[name](tool_input)


def _main() -> None:
    print("=== Testing web_search ===")
    print(web_search("What is the difference between RAG and fine-tuning in LLMs?"))

    print("\n=== Testing run_code ===")
    print(run_code("import math\nprint(math.sqrt(144))"))

    print("\n=== Testing run_code with error ===")
    print(run_code("print(1/0)"))

    print("\n=== Testing direct_answer ===")
    print(direct_answer("The answer is 42."))

    print("\n=== Testing execute_tool with unknown tool ===")
    print(execute_tool("unknown_tool", "input"))


if __name__ == "__main__":
    _main()
