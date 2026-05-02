import time
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
model = os.getenv("OPENAI_MODEL_DEEP", "gpt-5")

SYNTHESIZE_PROMPT = """
You are the synthesize phase of a stem agent.
You have received raw observations about how experts work in a domain.

Your job is to transform those observations into a concrete agent configuration.

You must return ONLY a valid JSON object with this exact structure:
{
    "domain": "the domain name",
    "identity": "a one-sentence description of what this agent is and does",
    "system_prompt": "the full system prompt for the specialized agent",
    "tools": ["list", "of", "tool", "names", "the", "agent", "needs"],
    "accepts": ["list of task types this agent accepts"],
    "rejects": ["list of task types this agent rejects"],
    "stop_criteria": "how the agent knows when it has a good enough answer",
    "quality_criteria": "what makes a good result in this domain"
}

The system_prompt must be detailed and specific. It should encode the expert knowledge from the observations.
Do not include any text outside the JSON object.
"""


def synthesize(domain: str, observations: dict) -> dict:
    print(f"[STEM/synthesize] Synthesizing agent configuration for '{domain}'...")

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIZE_PROMPT},
            {
                "role": "user",
                "content": f"Domain: {domain}\n\nObservations:\n{observations['content']}",
            },
        ],
    )
    elapsed = time.time() - start

    raw = response.choices[0].message.content.strip()

    try:
        config = json.loads(raw)
        print(
            f"[STEM/synthesize] Done in {elapsed:.2f}s. Agent configuration generated.\n"
        )
    except json.JSONDecodeError:
        print(
            "[STEM/synthesize] Warning: could not parse JSON. Returning raw output.\n"
        )
        config = {"raw": raw}

    return {"config": config, "model": model, "elapsed": elapsed}


if __name__ == "__main__":
    from observe import observe

    observations = observe("Technical Research")
    result = synthesize("Technical Research", observations)
    print(json.dumps(result["config"], indent=2))
