---
title: Tool Selection Accuracy
sidebar:
    order: 1
---

### Definitions

**Tool Selection Accuracy** measures how well an LLM selects a tool / function in a given module.

The used tools are compared with the expected tools and the metric outputs:
-   `num_correct`: total number of tools that are selected AND called with the correct arguments
-   `score`: `num_correct` / total number of tools in `ground_truth`

<br>


### Example Usage

Required data items: `tools`, `ground_truths`

```python
from continuous_eval.metrics import ToolSelectionAccuracy

tools = [
    FunctionTool.from_defaults(fn=useless, name=f"useless"),
    FunctionTool.from_defaults(fn=multiply, name="multiply"),
]

ground_truths = [
    FunctionTool.from_defaults(fn=multiply, name="multiply"),
    FunctionTool.from_defaults(fn=add, name="add"),
]

datum = {
    "tools": tools,
    "ground_truths": ground_truths,
},

metric = ToolSelectionAccuracy()
print(metric(**datum))
```

### Example Output

```JSON
{
    "num_correct": 1, 
    "score": 0.5
}
```
