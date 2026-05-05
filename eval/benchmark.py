"""
eval/benchmark.py

Evaluation benchmark for the stem agent vs. baseline comparison.

Runs a fixed set of tasks through both agents and compares:
- Steps used (efficiency)
- Domain rejection accuracy (specialization)
- Response quality (GPT-4o as judge)

Design decisions:
- GPT-4o as judge: automated quality evaluation without human annotation.
  The judge receives both responses without knowing which is which to
  avoid bias. This is a known technique in LLM evaluation.
- Fixed task set: same tasks for both agents ensures a fair comparison.
  Tasks are split into in-domain and out-of-domain to measure both
  quality and classification accuracy.
- Judge uses structured outputs: guarantees a parseable score on every
  evaluation without fragile JSON parsing.
"""

import json
import os
import time
from pathlib import Path
from typing import TypedDict

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from agent.react import run
from eval.baseline import BASELINE_CONFIG
from core.logger import get_logger
from stem.generate import load_existing

load_dotenv()

log = get_logger(__name__)
client = OpenAI(max_retries=3)

_judge_model = os.getenv("OPENAI_MODEL_FAST", "gpt-4o")

# Tasks split into in-domain and out-of-domain.
# In-domain tasks should be accepted and answered by the stem agent.
# Out-of-domain tasks should be rejected by the stem agent but accepted
# by the baseline (which has no classifier).
_IN_DOMAIN_TASKS = [
    "What are the main differences between PostgreSQL and SQLite for read-heavy workloads?",
    "What is the current state of the art in retrieval-augmented generation for question answering?",
    "What are the trade-offs between transformer-based models and RNNs for sequence modeling tasks?",
    "Write and execute Python code that generates 100 random numbers from a normal distribution and computes their mean and standard deviation.",
    "Search for the time complexity of binary search. Then write and execute Python code that measures how many comparisons it takes to find a value in a sorted list of 1000 elements.",
]

_OUT_OF_DOMAIN_TASKS = [
    "Write unit tests for a Python function that sorts a list.",
    "Give me a recipe for chocolate chip cookies.",
    "Write a short story about a robot who learns to paint.",
    "Should I incorporate my startup as an LLC or a C-Corp?",
    "Build me a React component for a login form with email and password fields.",
]

_COMPLEX_TASKS = [
    (
        "Experimental design",
        "I want to compare the inference latency of BERT-base vs DistilBERT on a text classification task. "
        "Design a rigorous benchmarking methodology. Include: metrics to measure, how to control for confounders, "
        "minimum number of runs, and how to report uncertainty.",
    ),
    (
        "Research critique",
        "A paper claims their new model achieves 94.2% accuracy on GLUE, beating the previous SOTA of 93.8%. "
        "What questions would you ask to evaluate whether this claim is valid and reproducible?",
    ),
    (
        "Trade-off analysis",
        "We need to choose between fine-tuning a small LLM vs using RAG for a customer support chatbot. "
        "Frame this as a research question, define success criteria, identify the key variables to measure, "
        "and propose a minimal experiment to inform the decision.",
    ),
]


class JudgeScore(BaseModel):
    """Structured output for the LLM judge."""

    score: int  # 1-5 quality score
    reasoning: str  # Brief explanation of the score


class TaskResult(TypedDict):
    """Result for a single task run."""

    task: str
    answer: str
    steps: int
    accepted: bool
    elapsed: float
    quality_score: int | None


class BenchmarkResult(TypedDict):
    """Aggregated benchmark results for one agent."""

    agent: str
    in_domain_results: list[TaskResult]
    out_of_domain_results: list[TaskResult]
    avg_steps: float
    avg_quality: float
    rejection_accuracy: float
    false_acceptance_rate: float
    complex_results: list[TaskResult]
    avg_quality_complex: float
    total_elapsed: float


_JUDGE_PROMPT = """
You are an impartial judge evaluating the quality of a technical research agent's response.

Task: {task}
Response: {response}

Rate the response on a scale of 1-5:
1 - Completely wrong or irrelevant
2 - Partially correct but missing key information
3 - Correct but superficial
4 - Correct and reasonably detailed
5 - Excellent — accurate, detailed, and actionable

Be strict. Only give a 5 if the response is genuinely excellent.
"""


def _judge_response(task: str, response: str) -> int:
    """
    Use GPT-4o to evaluate the quality of a response.

    Args:
        task: The original task.
        response: The agent's response to evaluate.

    Returns:
        Quality score from 1 to 5.
    """
    try:
        result = client.beta.chat.completions.parse(
            model=_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": _JUDGE_PROMPT.format(task=task, response=response),
                }
            ],
            response_format=JudgeScore,
        )
        return result.choices[0].message.parsed.score
    except openai.OpenAIError as e:
        log.warning("Judge failed for task '%s': %s", task[:50], e)
        return 0


def _run_task(task: str, agent_config: dict, judge: bool = True) -> TaskResult:
    """
    Run a single task through an agent and optionally judge the response.

    Args:
        task: The task string to execute.
        agent_config: Agent configuration dict.
        judge: Whether to evaluate response quality with the LLM judge.

    Returns:
        TaskResult with answer, steps, acceptance, timing, and quality score.
    """
    start = time.time()
    result = run(task, agent_config)
    elapsed = time.time() - start

    quality_score = None
    if judge and result["accepted"]:
        quality_score = _judge_response(task, result["answer"])

    return TaskResult(
        task=task,
        answer=result["answer"],
        steps=result["steps"],
        accepted=result["accepted"],
        elapsed=elapsed,
        quality_score=quality_score,
    )


def _run_benchmark(agent_config: dict, agent_name: str) -> BenchmarkResult:
    """
    Run the full benchmark for a single agent.

    Args:
        agent_config: Agent configuration dict.
        agent_name: Display name for logging.

    Returns:
        BenchmarkResult with aggregated metrics.
    """
    log.info("Running benchmark for agent: %s", agent_name)
    start = time.time()

    in_domain_results = []
    for task in _IN_DOMAIN_TASKS:
        log.info("  Task: %s...", task[:60])
        result = _run_task(task, agent_config)
        in_domain_results.append(result)
        log.info(
            "  → accepted=%s steps=%d quality=%s",
            result["accepted"],
            result["steps"],
            result["quality_score"],
        )

    out_of_domain_results = []
    for task in _OUT_OF_DOMAIN_TASKS:
        log.info("  Task: %s...", task[:60])
        result = _run_task(task, agent_config, judge=False)
        out_of_domain_results.append(result)
        log.info(
            "  → accepted=%s steps=%d",
            result["accepted"],
            result["steps"],
        )

    complex_results = []
    for label, task in _COMPLEX_TASKS:
        log.info("  Complex task: %s...", label)
        result = _run_task(task, agent_config)
        complex_results.append(result)
        log.info(
            "  → accepted=%s steps=%d quality=%s",
            result["accepted"],
            result["steps"],
            result["quality_score"],
        )

    # Metrics
    accepted_in = [r for r in in_domain_results if r["accepted"]]
    avg_steps = (
        sum(r["steps"] for r in accepted_in) / len(accepted_in) if accepted_in else 0.0
    )

    scores = [r["quality_score"] for r in in_domain_results if r["quality_score"]]
    avg_quality = sum(scores) / len(scores) if scores else 0.0

    # Rejection accuracy: out-of-domain tasks that were correctly rejected
    # Rejection accuracy: out-of-domain tasks correctly rejected
    correctly_rejected = sum(1 for r in out_of_domain_results if not r["accepted"])
    rejection_accuracy = correctly_rejected / len(_OUT_OF_DOMAIN_TASKS)

    # False acceptance rate: out-of-domain tasks incorrectly accepted
    false_acceptance_rate = 1 - rejection_accuracy

    complex_scores = [r["quality_score"] for r in complex_results if r["quality_score"]]
    avg_quality_complex = (
        sum(complex_scores) / len(complex_scores) if complex_scores else 0.0
    )

    total_elapsed = time.time() - start

    log.info(
        "Done. avg_steps=%.1f avg_quality=%.2f rejection_accuracy=%.0f%%",
        avg_steps,
        avg_quality,
        rejection_accuracy * 100,
    )

    return BenchmarkResult(
        agent=agent_name,
        in_domain_results=in_domain_results,
        out_of_domain_results=out_of_domain_results,
        avg_steps=avg_steps,
        avg_quality=avg_quality,
        rejection_accuracy=rejection_accuracy,
        false_acceptance_rate=false_acceptance_rate,
        complex_results=complex_results,
        avg_quality_complex=avg_quality_complex,
        total_elapsed=total_elapsed,
    )


def _print_comparison(stem: BenchmarkResult, baseline: BenchmarkResult) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Baseline':>12} {'Stem Agent':>12}")
    print("-" * 56)
    print(
        f"{'Avg steps (in-domain)':<30} {baseline['avg_steps']:>12.1f} {stem['avg_steps']:>12.1f}"
    )
    print(
        f"{'Avg quality score (1-5)':<30} {baseline['avg_quality']:>12.2f} {stem['avg_quality']:>12.2f}"
    )
    print(
        f"{'Rejection accuracy':<30} {baseline['rejection_accuracy']:>11.0%} {stem['rejection_accuracy']:>11.0%}"
    )
    print(
        f"{'False acceptance rate':<30} {baseline['false_acceptance_rate']:>11.0%} {stem['false_acceptance_rate']:>11.0%}"
    )
    print(
        f"{'Avg quality complex (1-5)':<30} {baseline['avg_quality_complex']:>12.2f} {stem['avg_quality_complex']:>12.2f}"
    )
    print(
        f"{'Total elapsed (s)':<30} {baseline['total_elapsed']:>12.1f} {stem['total_elapsed']:>12.1f}"
    )

    print("\n--- In-domain task breakdown ---")
    for s, b in zip(stem["in_domain_results"], baseline["in_domain_results"]):
        print(f"\nTask: {s['task'][:70]}")
        print(f"  Baseline  → steps={b['steps']} quality={b['quality_score']}")
        print(f"  Stem      → steps={s['steps']} quality={s['quality_score']}")

    print("\n--- Out-of-domain rejection ---")
    for s, b in zip(stem["out_of_domain_results"], baseline["out_of_domain_results"]):
        print(f"\nTask: {s['task'][:70]}")
        print(f"  Baseline  → accepted={b['accepted']}")
        print(f"  Stem      → accepted={s['accepted']}")

    print("\n--- Complex task breakdown ---")
    for (label, _), s, b in zip(
        _COMPLEX_TASKS, stem["complex_results"], baseline["complex_results"]
    ):
        print(f"\nTask: {label}")
        print(f"  Baseline  → steps={b['steps']} quality={b['quality_score']}")
        print(f"  Stem      → steps={s['steps']} quality={s['quality_score']}")


def _main() -> None:
    domain = os.getenv("STEM_DOMAIN", "Technical Research")
    n_runs = int(os.getenv("BENCHMARK_RUNS", "3"))

    stem_config = load_existing(domain)
    if not stem_config:
        raise ValueError(
            f"No config found for domain '{domain}'. "
            "Run 'python main.py' first to generate the stem agent config."
        )

    all_stem = []
    all_baseline = []

    for i in range(1, n_runs + 1):
        log.info("Run %d/%d", i, n_runs)
        all_stem.append(_run_benchmark(stem_config, "stem-agent"))
        all_baseline.append(_run_benchmark(BASELINE_CONFIG, "baseline"))

    # Average across runs
    def _avg(results: list[BenchmarkResult], key: str) -> float:
        return sum(r[key] for r in results) / len(results)

    stem_avg = {
        "agent": "stem-agent",
        "avg_steps": _avg(all_stem, "avg_steps"),
        "avg_quality": _avg(all_stem, "avg_quality"),
        "rejection_accuracy": _avg(all_stem, "rejection_accuracy"),
        "false_acceptance_rate": _avg(all_stem, "false_acceptance_rate"),
        "total_elapsed": _avg(all_stem, "total_elapsed"),
        "in_domain_results": all_stem[-1]["in_domain_results"],
        "out_of_domain_results": all_stem[-1]["out_of_domain_results"],
        "complex_results": all_stem[-1]["complex_results"],
        "avg_quality_complex": _avg(all_stem, "avg_quality_complex"),
    }

    baseline_avg = {
        "agent": "baseline",
        "avg_steps": _avg(all_baseline, "avg_steps"),
        "avg_quality": _avg(all_baseline, "avg_quality"),
        "rejection_accuracy": _avg(all_baseline, "rejection_accuracy"),
        "false_acceptance_rate": _avg(all_baseline, "false_acceptance_rate"),
        "total_elapsed": _avg(all_baseline, "total_elapsed"),
        "in_domain_results": all_baseline[-1]["in_domain_results"],
        "out_of_domain_results": all_baseline[-1]["out_of_domain_results"],
        "complex_results": all_stem[-1]["complex_results"],
        "avg_quality_complex": _avg(all_stem, "avg_quality_complex"),
    }

    _print_comparison(stem_avg, baseline_avg)

    output_path = Path("config/generated/benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": n_runs,
                "stem": stem_avg,
                "baseline": baseline_avg,
                "raw_stem": all_stem,
                "raw_baseline": all_baseline,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    _main()
