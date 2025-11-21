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

## Data Structures

Roughly:

```python
from dataclasses import dataclass
from typing import Optional

Score = int  # 1..10


@dataclass
class WorkerOutput:
    worker_id: str
    result: str          # worker's output


@dataclass
class Judgment:
    worker_id: str
    result: str          # worker's output (copied here for convenience)
    score: Score         # 1..10
    review: str          # short critique from judge


@dataclass
class Task:
    initial_task: str               # original user task
    prev_best_result: Optional[str] = None
    prev_best_review: Optional[str] = None
    iteration: int = 0
```

- Task.initial_task never changes.
- prev_best_result and prev_best_review are filled in after the first iteration, so workers can try to beat the previous best.
- iteration is just a counter.

### Core Loop

High-level interface:

```python
from typing import List, Tuple
import random

MIN_SCORE: Score = 8         # "good enough" threshold (1..10)
MAX_ITER: int = 10           # maximum iterations


def run_parliament(initial_task: str) -> str:
    """
    Entry point. Returns the best result as a string.
    """
    task = Task(initial_task=initial_task)
    return _iterate(task)


def _iterate(task: Task) -> str:
    # 1) Workers generate results
    worker_outputs: List[WorkerOutput] = run_workers(task)

    # 2) Judges evaluate each worker's result
    judgments: List[Judgment] = run_judges(worker_outputs)

    # 3) Find all best workers according to average judge score
    best_worker_ids, best_avg_score = find_best_workers(judgments)

    # Randomly select one best judgment (worker + judge) among all best workers
    best_judgment = select_random_best_judgment(judgments, best_worker_ids)

    # 4) Stopping criteria
    if best_avg_score >= MIN_SCORE or task.iteration >= MAX_ITER:
        # Return the result of the randomly selected best judgment
        return best_judgment.result

    # 5) Prepare next iteration task with feedback from the best judgment
    next_task = Task(
        initial_task=task.initial_task,
        prev_best_result=best_judgment.result,
        prev_best_review=best_judgment.review,
        iteration=task.iteration + 1,
    )

    return _iterate(next_task)

Where you implement:

def run_workers(task: Task) -> List[WorkerOutput]:
    """
    Call all worker models with the same prompt (task + optional previous best)
    and return their outputs.
    """
    raise NotImplementedError


def run_judges(worker_outputs: List[WorkerOutput]) -> List[Judgment]:
    """
    For each worker output, call all judges to score and review it.
    Returns a flat list of Judgments.
    """
    raise NotImplementedError
```

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

Generalized:

```python
from collections import defaultdict


def find_best_workers(judgments: List[Judgment]) -> Tuple[List[str], float]:
    """
    Returns (best_worker_ids, best_avg_score).

    - best_worker_ids: all worker IDs whose average score is equal
      to the maximum average score.
    - best_avg_score: that maximum average score.
    """
    scores_by_worker = defaultdict(list)
    for j in judgments:
        scores_by_worker[j.worker_id].append(j.score)

    if not scores_by_worker:
        raise RuntimeError("No judgments provided")

    avg_score_by_worker = {
        worker_id: sum(scores) / len(scores)
        for worker_id, scores in scores_by_worker.items()
        if scores
    }

    best_avg_score = max(avg_score_by_worker.values())
    best_worker_ids = [
        worker_id
        for worker_id, avg_score in avg_score_by_worker.items()
        if avg_score == best_avg_score
    ]

    return best_worker_ids, best_avg_score
```

Now we have all best workers according to judges (there may be more than one).

### Random Selection Among Best Results

We want:
	•	If multiple workers have the same best average score, we treat all of them as “best workers”.
	•	We then pick one best judgment at random across all best workers.
That judgment’s result is the final output, and its review is what we feed into the next iteration (if any).

```python
def select_random_best_judgment(
    judgments: List[Judgment],
    best_worker_ids: List[str],
) -> Judgment:
    """
    Among all judgments belonging to the best workers, pick one at random.
    """
    candidates = [j for j in judgments if j.worker_id in best_worker_ids]
    if not candidates:
        raise RuntimeError("No judgments for best workers")
    return random.choice(candidates)
```

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
calls_per_iteration ≈ NUM_WORKERS           # worker calls
                      + NUM_WORKERS * NUM_JUDGES  # judge calls
total_calls ≈ calls_per_iteration * MAX_ITER
```

So it’s a good idea to start with:

- 2 workers,
- 1–2 judges,
- MAX_ITER = 3–5,

and only increase these numbers if you see real quality improvements.

### Summary

Parliament is a pattern, not a heavy framework:
- It separates generation and evaluation (workers vs judges).
- It uses several models to reduce single-model bias.
- It applies clear scoring and simple averaging to pick the best.
- It uses feedback + iteration so the system can actually improve across rounds instead of just re-rolling.

You can implement this in any stack; the code above is intentionally minimal so you can adapt it to your own infra and prompts.

