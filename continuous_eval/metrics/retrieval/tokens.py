import tiktoken

from continuous_eval.metrics.base import Metric

_CHARACTERS_PER_TOKEN = 4.0


class TokenCount(Metric):
    def __init__(self, encoder_name: str) -> None:
        super().__init__()
        if encoder_name == "approx":
            self._encoder = None
        else:
            try:
                self._encoder = tiktoken.get_encoding(encoder_name)
            except ValueError:
                raise ValueError(f"Invalid encoder name: {encoder_name}")

    def __call__(self, retrieved_context, **kwargs):
        ctx = "\n".join(retrieved_context)
        if self._encoder is None:
            num_tokens = int(len(ctx) / _CHARACTERS_PER_TOKEN)
        else:
            num_tokens = len(self._encoder.encode(ctx))
        return {"num_tokens": num_tokens}
