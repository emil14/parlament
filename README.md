# Parlament

This repo describes a pattern of utilizing several different LLMs to produce higher quality content.

It can be used for anything from book writing to code generation.

## Judges and Workers

We have 2 sets of AI agents:

1. Judges
2. Workers

Workers produce content and judges judge it.

Constraints:

- There can be as many judges and workers as we want
- All workers must share the same prompt but have different LLMs
- Same for judges

Example (pseudocode):

In this example all 3j and 3w are using the same models for simplicity.

```python
w1 = Worker(prompt="write a novel", model="gpt-5_1")
w2 = Worker(prompt="write a novel", model="gemini_3_pro")
w3 = Worker(prompt="write a novel", model="sonnet_4_5")

j1 = Judge(prompt="judge a novel", model="gpt-5_1")
j2 = Judge(prompt="judge a novel", model="gemini_3_pro")
j3 = Judge(prompt="judge a novel", model="sonnet_4_5")

p = Parlament(
  workers=[w1, w2, w3],
  judges=[j1, j2, j3],
)
```

