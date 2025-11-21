# Parliament

**Parliament** is a simple pattern for making several LLMs work together:

> Multiple **workers** generate answers → multiple **judges** evaluate them → the system iterates until we get a good enough result.

It’s useful for anything where quality really matters:

- long-form writing,
- non-trivial code generation,
- important emails, specs, docs, etc.

The key idea: instead of a single model trying to do everything, we separate **generation** (workers) from **evaluation** (judges), and we run several rounds where the next generation tries to *beat* the previous best answer.

---

## TL;DR

1. You have **N workers** (different models, same prompt) that generate answers.
2. You have **M judges** (different models, same prompt) that score each worker’s answer from **1 to 10**.
3. For each worker, you compute the **average judge score**.
4. If the best average score is **≥ MIN_SCORE** or you hit **MAX_ITER** (e.g. 10), you stop.
5. Otherwise, you feed the **best answer + review** back into the next iteration and say:  
   > “Here’s the previous best result and the critique. Try to beat it.”
6. If multiple workers tie for best average score, you pick **one at random** among them.

That’s it.

---

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

---

### Judges

- **Judges evaluate worker outputs.**
- All judges also share the **same judging prompt**.
- Each judge may use a different LLM (but typically a **stronger or equal** model than the workers).
- Judges score on a **fixed scale (1–10)** and optionally produce a short textual review.

Examples:

- Judge 1 → `gpt-4.1`  
- Judge 2 → `gpt-4.1-large`  

Judges decide *how good* each worker’s output is with respect to the task.

---

## Data Structures

Roughly:

```python
from dataclasses import dataclass
from typing import Optional

Score = int  # 1..10

@dataclass
class Judgment:
    worker_id: str
    result: str          # worker's output
    score: Score         # 1..10
    review: str          # short critique

@dataclass
class Task:
    initial_task: str              # original user task
    prev_best_result: Optional[str] = None
    prev_best_review: Optional[str] = None
    iteration: int = 0
