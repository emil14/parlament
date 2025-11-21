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
- Then every judge judges the work of every worker.
- If not a single result is good enough, a new iteration is performed.
- If there's more than one good enough results, random one is returned.
- If there's only one good enough result, it is returned as the best one.
- If maximum amount of iterations was performed, then best result is returned, and the randomly selected one if there's more than one best.




