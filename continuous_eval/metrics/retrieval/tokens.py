from typing import List, Union

import tiktoken

from continuous_eval.metrics.base import Field, Metric

_CHARACTERS_PER_TOKEN = 4.0


class TokenCount(Metric):
    """
    Count the number of tokens in the retrieved context.
    """

    # Encodings specify how text is converted into tokens.
    # Different models use different encodings.
    # | Encoding Name            | OpenAI Models                                                                 |
    # |--------------------------|-------------------------------------------------------------------------------|
    # | `o200k_base`             | `gpt-4o`, `gpt-4o-mini`                                                      |
    # | `cl100k_base`            | `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`, `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large` |
    # | `p50k_base`              | Codex models, `text-davinci-002`, `text-davinci-003`                         |
    # | `r50k_base` (or `gpt2`)  | GPT-3 models like `davinci`                                                  |

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

    def compute(self, retrieved_context: Union[str, List[str]], **kwargs):
        ctx = (
            "\n".join(retrieved_context)
            if isinstance(retrieved_context, list)
            else retrieved_context
        )
        if self._encoder is None:
            num_tokens = int(len(ctx) / _CHARACTERS_PER_TOKEN)
        else:
            num_tokens = len(self._encoder.encode(ctx))
        return {"num_tokens": num_tokens}

    @property
    def schema(self):
        return {"num_tokens": Field(type=int)}
