import subprocess
import sys
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
fast_model = os.getenv("OPENAI_MODEL_FAST", "gpt-4o")

# ─── web_search ───────────────────────────────────────────────

WEB_SEARCH_PROMPT = """
You are a web search assistant. The user gives you a search query.
Search for current, accurate information and return a concise summary.
Include specific facts, numbers, and sources where relevant.
Be direct and informative. Do not add commentary about the search process.
"""


def web_search(query: str) -> str:
    print(f"  [tool/web_search] Query: {query}")

    response = client.chat.completions.create(
        model=fast_model,
        messages=[
            {"role": "system", "content": WEB_SEARCH_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    result = response.choices[0].message.content
    return result


# ─── run_code ─────────────────────────────────────────────────


def run_code(code: str) -> str:
    print("  [tool/run_code] Executing code...")

    try:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output else "Code executed successfully. No output."
        else:
            return f"Error:\n{result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (15s limit)."
    except Exception as e:
        return f"Error: {str(e)}"


# ─── direct_answer ────────────────────────────────────────────


def direct_answer(answer: str) -> str:
    print("  [tool/direct_answer] Delivering final answer.")
    return answer


# ─── Tool registry ────────────────────────────────────────────

TOOLS = {"web_search": web_search, "run_code": run_code, "direct_answer": direct_answer}


def execute_tool(name: str, input: str) -> str:
    if name not in TOOLS:
        return f"Error: tool '{name}' does not exist. Available tools: {list(TOOLS.keys())}"
    return TOOLS[name](input)


# ─── Test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing web_search ===")
    print(web_search("What is the difference between RAG and fine-tuning in LLMs?"))

    print("\n=== Testing run_code ===")
    print(run_code("import math\nprint(math.sqrt(144))"))

    print("\n=== Testing run_code with error ===")
    print(run_code("print(1/0)"))

    print("\n=== Testing direct_answer ===")
    print(direct_answer("The answer is 42."))
