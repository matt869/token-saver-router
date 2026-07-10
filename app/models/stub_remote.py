"""Keyless stand-in for :class:`~app.models.remote_model.RemoteModel`.

Lets you demo the dashboard and metrics without a Fireworks key. Crucially it
echoes token usage **proportional to the prompt it actually receives** — and
the agent hands it the *already-compressed* prompt — so ``prompt_tokens_after``
tracks the real compressed size. Compression can then only move the metric in
the correct direction (after ≤ before); a constant stub could show tokens going
*up*, which is exactly the bug this avoids.

Interface-compatible with ``RemoteModel``: same ``model`` attribute and
``generate(prompt) -> RemoteResult`` with the usage split populated.
"""

from __future__ import annotations

from app.models.remote_model import RemoteResult
from app.tokens import count_tokens


class StubRemoteModel:
    """Deterministic, keyless RemoteModel stand-in for demos and tests."""

    def __init__(
        self,
        model: str = "stub/echo",
        answer: str = "(stub remote answer)",
        completion_tokens: int = 16,
    ):
        self.model = model
        self.answer = answer
        self.completion_tokens = max(0, completion_tokens)

    def generate(self, prompt: str, max_new_tokens=None) -> RemoteResult:
        prompt_tokens = count_tokens(prompt)  # real count of the compressed prompt
        return RemoteResult(
            text=self.answer,
            tokens=prompt_tokens + self.completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=self.completion_tokens,
        )
