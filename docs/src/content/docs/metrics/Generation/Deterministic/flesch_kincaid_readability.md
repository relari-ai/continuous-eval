---
title: Flesch–Kincaid Readability
sidebar:
    order: 2
---

### Definitions

The Flesch–Kincaid measures how easy it is to understand an by considering factors like sentence length and word complexity. There are two main types: the Flesch Reading Ease, which rates texts on a scale from easy to difficult, and the Flesch–Kincaid Grade Level, which estimates the U.S. school grade level needed to comprehend the text.

**Flesch Reading Ease** higher scores indicate material that is easier to read; lower numbers mark passages that are more difficult to read.

**Flesch–Kincaid Grade Level** corresponds to a U.S. grade level. For example, a score of 8.0 suggests that the text should be understandable by an 8th grader. Higher scores indicate material that is more complex and thus requires a higher level of education to understand.

:::note
**Negative scores.** Both scores can be negative.
A negative Flesh reading ease indicates that the text is very difficult to read and a negative Flesch–Kincaid grade level indicates that the text is very easy to read (they are inversely correlated).
Note that the lowest grade level score in theory is −3.40.
:::

To know more about the test read [Wikipedia](https://en.wikipedia.org/wiki/Flesch–Kincaid_readability_tests).

### Example Usage

Required data items: `answer`

```python
from continuous_eval.metrics.generation.text  import FleschKincaidReadability

datum = {
    "answer": "The cat sat on the mat.",
}

metric = FleschKincaidReadability()
print(metric(**datum))
```

### Example Output

```JSON
{
    "flesch_reading_ease": 116.14500000000001,
    "flesch_kincaid_grade_level": -1.4499999999999993,
}
```
