---
title: Tool Selection Accuracy
sidebar:
    order: 1
---

### Definitions

**Tool Selection Accuracy** measures how well an LLM selects a tool / function in a given module.

The used tools are compared with the expected tools and the metric outputs:

- `num_correct`: total number of tools that are selected AND called with the correct arguments
- `score`: `num_correct` / total number of tools in `ground_truths`

### Example Usage

Required data items: `tools`, `ground_truths`

```python
from continuous_eval.metrics.tools.match import ToolSelectionAccuracy
from continuous_eval.eval.types import ToolCall

tools = [
    ToolCall(name="useless", kwargs={}),
    ToolCall(name="multiply", kwargs={"a": 2, "b": 3}),
]

ground_truths = [
    ToolCall(name="useless", kwargs={}),
    ToolCall(name="add", kwargs={"a": 2, "b": 3}),
]

datum = {
    "tools": tools,
    "ground_truths": ground_truths,
}

metric = ToolSelectionAccuracy()
print(metric(**datum))
```

### Example Output

```python
{
    "num_correct": 1, 
    "score": 0.5
}
```
