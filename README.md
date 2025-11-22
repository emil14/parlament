# Parliament

**Parliament** is a simple pattern for making several LLMs work together:

> Multiple **workers** generate answers → multiple **judges** evaluate them → the system iterates until we get a good enough result.

It’s useful for anything where quality really matters:

- long-form writing,
- non-trivial code generation,
- important emails, specs, docs, etc.

The key idea: instead of a single model trying to do everything, we separate **generation** (workers) from **evaluation** (judges), and we run several rounds where the next generation tries to *beat* the previous best answer.

## TL;DR

1. You have **N workers** (different models, same prompt) that generate answers.
2. You have **M judges** (different models, same judging prompt) that score each worker’s answer from **1 to 10**.
3. For each worker, you compute the **average judge score**.
4. If the best average score is **≥ MIN_SCORE** or you hit **MAX_ITER** (e.g. 10), you stop.
5. Otherwise, you feed the **best answer + review** back into the next iteration and say:  
   > “Here’s the previous best result and the critique. Try to beat it.”
6. If multiple workers tie for the best average score, you **randomly select one of the best results** among them.

## Concepts

### Workers

- **Workers produce content.**
- All workers share the **same prompt** (same task description).
- Each worker uses a **different LLM** (or at least different settings).
- This way you’re mostly comparing **model behavior**, not prompt differences.

Examples:

- Worker 1 → `gpt-4.1-mini`  
- Worker 2 → `gpt-4.1`  
- Worker 3 → `some-open-source-model`

All of them see the same task, but may produce different outputs.

### Judges

- **Judges evaluate worker outputs.**
- All judges share the **same judging prompt**.
- Each judge may use a different LLM, typically **equal or stronger** than the worker models (since they’re making the decisions).
- Judges score on a **fixed scale (1–10)** and optionally produce a short textual review.

Examples:

- Judge 1 → `gpt-4.1`  
- Judge 2 → `gpt-4.1-large`  

Judges decide *how good* each worker’s output is with respect to the task.

### Best Result Selection

For N workers and M judges, we have N * M individual judgments.

Example: 2 workers (w1, w2) and 2 judges (j1, j2) → we get 4 judgments:

- j1-w1 (judge 1 scoring worker 1)
- j1-w2
- j2-w1
- j2-w2

For each worker, we compute the average of all judge scores for that worker:

w1_avg = (score_j1_w1 + score_j2_w1) / 2
w2_avg = (score_j1_w2 + score_j2_w2) / 2

### Random Selection Among Best Results

We want:
	•	If multiple workers have the same best average score, we treat all of them as “best workers”.
	•	We then pick one best judgment at random across all best workers.
That judgment’s result is the final output, and its review is what we feed into the next iteration (if any).

This matches the idea:

In case we have more than one best result (several results with the highest score), we randomly select one of them.

### How Iteration Leads to Improvement

In each iteration, workers receive not only the original task, but also the previous best result and its critique.

Example worker prompt (simplified):

```
You are a worker model in a multi-model "parliament" system.

Your task is:

{{ initial_task }}

We previously had the best result:

[Previous best result]
{{ prev_best_result }}

Judges gave this critique:

[Previous best review]
{{ prev_best_review }}

Your job is to produce a NEW answer that:
- strictly improves on the previous best where it was weak,
- keeps the parts that are already strong,
- and stays fully aligned with the original task.

Return ONLY your final answer, no explanations.

On the first iteration:
	•	prev_best_result and prev_best_review are None → workers just solve the task from scratch.

From the second iteration onward:
	•	Workers are explicitly in “beat the previous best” mode.
```

## Naive Implementation

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import defaultdict
import random

Score = int  # 1..10


@dataclass
class WorkerOutput:
    worker_id: str
    answer: str  # worker's answer


@dataclass
class Evaluation:
    worker_id: str
    answer: str          # same answer as judged
    score: Score         # 1..10
    review: str          # short critique from evaluator


@dataclass
class IterationState:
    task_prompt: str
    best_answer: Optional[str] = None
    best_answer_review: Optional[str] = None
    index: int = 0


QUALITY_THRESHOLD: Score = 8
MAX_ITERATIONS: int = 10


def run_parliament(task_prompt: str) -> str:
    """
    Entry point. Returns the best answer as a string.
    """
    state = IterationState(task_prompt=task_prompt)
    return _iterate(state)


def _iterate(state: IterationState) -> str:
    # 1) Workers (candidates) generate answers
    worker_outputs: List[WorkerOutput] = run_workers(state)

    # 2) Evaluators (judges) score each worker's answer
    evaluations: List[Evaluation] = run_evaluators(worker_outputs)

    # 3) Find the best worker according to average evaluator score
    best_evaluation, best_avg_score = find_best_worker(evaluations)

    # 4) Stopping criteria
    if best_avg_score >= QUALITY_THRESHOLD or state.index >= MAX_ITERATIONS:
        return best_evaluation.answer

    # 5) Prepare next iteration state with feedback from the best evaluation
    next_state = IterationState(
        task_prompt=state.task_prompt,
        best_answer=best_evaluation.answer,
        best_answer_review=best_evaluation.review,
        index=state.index + 1,
    )

    return _iterate(next_state)


def run_workers(state: IterationState) -> List[WorkerOutput]:
    """
    Call all worker models (candidates) with the same prompt
    (task + best-answer feedback if available) and return their outputs.
    """
    raise NotImplementedError


def run_evaluators(worker_outputs: List[WorkerOutput]) -> List[Evaluation]:
    """
    For each worker output, call all evaluator models to score and review it.
    Returns a flat list of Evaluation objects.
    """
    raise NotImplementedError


def find_best_worker(evaluations: List[Evaluation]) -> Tuple[Evaluation, float]:
    """
    Given all evaluations (N workers * M evaluators), compute the average score
    per worker and return:

    - one randomly chosen Evaluation for a worker with the best average score
    - the best average score itself

    If multiple workers tie for the best average score, we select one of them
    at random.
    """
    if not evaluations:
        raise RuntimeError("No evaluations provided")

    scores_by_worker = defaultdict(list)
    for e in evaluations:
        scores_by_worker[e.worker_id].append(e.score)

    if not scores_by_worker:
        raise RuntimeError("No scores aggregated per worker")

    avg_score_by_worker = {
        worker_id: sum(scores) / len(scores)
        for worker_id, scores in scores_by_worker.items()
        if scores
    }

    if not avg_score_by_worker:
        raise RuntimeError("No average scores computed")

    best_avg_score = max(avg_score_by_worker.values())
    best_worker_ids = [
        worker_id
        for worker_id, avg_score in avg_score_by_worker.items()
        if avg_score == best_avg_score
    ]

    # All evaluations belonging to workers with the best average score
    candidate_evaluations = [
        e for e in evaluations if e.worker_id in best_worker_ids
    ]
    if not candidate_evaluations:
        raise RuntimeError("No evaluations for best-scoring workers")

    best_evaluation = random.choice(candidate_evaluations)
    return best_evaluation, best_avg_score
```

### Configuration

Suggested defaults:

```python
MIN_SCORE = 8     # 1..10 scale – 8+ is "good enough"
MAX_ITER = 10     # up to 10 iterations
NUM_WORKERS = 2   # start small
NUM_JUDGES = 2
```

You can tune this per domain:
- For trivial tasks, set MAX_ITER = 1.
- For very hard tasks, you can:
- increase NUM_WORKERS instead of cranking iterations,
- or raise MIN_SCORE if you need very strict quality.

### Cost & Practical Notes

The total number of LLM calls grows roughly like:

```python
calls_per_iteration ≈ NUM_WORKERS * NUM_JUDGES
total_calls ≈ calls_per_iteration * MAX_ITER
```

### Summary

Parliament is a pattern, not a heavy framework:
- It separates generation and evaluation (workers vs judges).
- It uses several models to reduce single-model bias.
- It applies clear scoring and simple averaging to pick the best.
- It uses feedback + iteration so the system can actually improve across rounds instead of just re-rolling.

You can implement this in any stack; the code above is intentionally minimal so you can adapt it to your own infra and prompts.

