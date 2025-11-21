# Parlament

I've often thought "If only I could force 3-4 LLMs to work on task together so they can then magically decide who did it better, or maybe somehow merge their results into a single one without me doing anything", so I've created this repo.

It describes a pattern of utilizing several different LLMs to produce higher quality content, that can be used for anything from book writing to code generation.

## Process

An iterative generational process that takes a task, processes it and returns result.

It runs several iterations under the hood, until result considered good enough.

## Judges and Workers

We have 2 sets of AI agents:

1. Judges
2. Workers

Workers produce content and judges judge it.

Separation between them exist to reduce responsibility of the prompts.

- There can be as many judges and workers as we want
- All workers must share the same prompt but have different LLMs
- Same for judges

## Loop

- Workers produce their results.
- Every judge judges the work of every worker.
- If not a single result is good enough, a new iteration is performed.
- If there's more than one good enough results, random one is returned.
- If there's only one good enough result, it is returned as the best one.
- If maximum amount of iterations was performed, then best result is returned, and the randomly selected one if there's more than one best.

```python
def iter(task: Task, i: int = 0):
  results: list[str] = workers(task)
  judgments: list[Judgment] = judges(results)
  best: list[Judgment] = find_best(judgments)

  if best.score >= min_score or i == max_iter:
    return best.result

  next_task = Task(initial_task=task, prev_judgment=best)
  return iter(next_task, i+1)
```

The data structures are roughly this:

```python
class Task:
  initial_task: str
  prev_judgment: Judgment | None

class Judgment:
  result: str
  score: int
  review: str
```

## Best Result

For `n` workers and `m` judges we have `n*m` judgments.

For example for 2 workers and 2 judges we have 4 judgments: `j1-w1, j1-w2, j2-w1, j2-j2`

The avg score for `w1` is `(j1-w1 / j2-w1) / 2` and for `w2` is `(j1-w2 / j2-w2) / 2`.

If `w1_avg` is higher than `w2_avg` then we return `w1_avg`, otherwise `w2_avg`.

If `w1_avg` and `w2_avg` are equal, we randomly select one of them.

Same rules apply for >2 results.



