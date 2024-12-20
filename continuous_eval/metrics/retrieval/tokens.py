from typing import List

import tiktoken

from continuous_eval.metrics.base import Arg, Field, Metric

_CHARACTERS_PER_TOKEN = 4.0


class TokenCount(Metric):
    def __init__(self, encoder_name: str = "gpt-4o-mini") -> None:
        super().__init__(is_cpu_bound=True)
        if encoder_name == "approx":
            self._encoder = None
        else:
            try:
                self._encoder = tiktoken.get_encoding(encoder_name)
            except ValueError:
                try:
                    self._encoder = tiktoken.encoding_for_model(encoder_name)
                except ValueError:
                    raise ValueError(
                        f"Invalid encoder name: {encoder_name}. You can use encoders names like `o200k_base` or model names like `gpt4o-mini`."
                    )

    def compute(self, retrieved_context, **kwargs):
        ctx = "\n".join(retrieved_context)
        if self._encoder is None:
            num_tokens = int(len(ctx) / _CHARACTERS_PER_TOKEN)
        else:
            num_tokens = len(self._encoder.encode(ctx))
        return {"num_tokens": num_tokens}

    @property
    def args(self):
        return {
            "retrieved_context": Arg(type=List[str], is_ground_truth=False),
        }

    @property
    def schema(self):
        return {
            "num_tokens": Field(type=int),
        }
