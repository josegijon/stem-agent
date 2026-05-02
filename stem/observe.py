import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("OPENAI_MODEL")

if not model:
    raise ValueError("OPENAI_MODEL not set in .env")

client = OpenAI()

OBSERVE_PROMPT = """
You are the observe phase of a stem agent.
Your job is to research how experts actually work in a given domain.

Research and describe:
1. What steps do experts follow when working in this domain?
2. What tools do they use?
3. What makes a good result in this domain?
4. What are common mistakes or pitfalls?

Be specific and practical. You are gathering raw material for another agent to synthesize.
"""


def observe(domain: str) -> str:
    print(f"[STEM/observe] Searching for how '{domain}' works in practice...")

    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": OBSERVE_PROMPT},
            {
                "role": "user",
                "content": f"Research how experts approach and execute tasks in this domain: {domain}",
            },
        ],
    )

    elapsed = time.time() - start
    result = response.choices[0].message.content

    print(
        f"[STEM/observe] Done in {elapsed:.2f}s. Gathered {len(result)} characters.\n"
    )

    return {
        "content": result,
        "model": model,
        "elapsed": elapsed,
        "characters": len(result),
    }


if __name__ == "__main__":
    result = observe("Technical Research")
    print(result["content"])
