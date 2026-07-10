"""Remote model wrapper (scored path): Fireworks AI.

Calls the Fireworks OpenAI-compatible chat-completions endpoint with ``requests``
and returns both the answer text and the *total* tokens billed. Those tokens are
the only ones that count toward the hackathon score, so the agent tries hard to
avoid this path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests


class RemoteModelError(RuntimeError):
    """Raised when the Fireworks API call fails or returns no usable answer."""


@dataclass
class RemoteResult:
    """Answer plus the token usage billed by the remote provider (scored).

    ``prompt_tokens``/``completion_tokens`` are the provider's own usage split —
    real billing data, not an estimate — so the metrics layer can report
    "prompt tokens actually sent" per request.
    """

    text: str
    tokens: int  # usage.total_tokens reported by Fireworks (prompt + completion)
    prompt_tokens: int = 0  # usage.prompt_tokens (what the optimized prompt cost)
    completion_tokens: int = 0  # usage.completion_tokens


class RemoteModel:
    """Thin Fireworks AI chat-completions client."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        max_new_tokens: int = 256,
        timeout: int = 60,
        retries: int = 1,
        backoff: float = 0.5,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff = backoff

    def _post_with_retry(self, url: str, payload: dict, headers: dict) -> dict:
        """POST with retry on *transient* failures only.

        Retryable: connection errors/timeouts (no status) and 429/5xx. A 4xx
        like 401/400 will not heal on retry, so it fails immediately. Without
        this, a single transient 429 would push a HARD query into local
        failover — trading answer quality away to save one retry.
        """

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                try:
                    return resp.json()
                except ValueError as exc:  # invalid JSON
                    raise RemoteModelError(f"Fireworks returned invalid JSON: {exc}") from exc
            except requests.RequestException as exc:  # network / HTTP error
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                last_exc = exc
                if not retryable or attempt == self.retries:
                    raise RemoteModelError(f"Fireworks request failed: {exc}") from exc
                time.sleep(self.backoff * (attempt + 1))
        # Unreachable, but keeps type-checkers honest.
        raise RemoteModelError(f"Fireworks request failed: {last_exc}") from last_exc

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> RemoteResult:
        if not self.api_key:
            raise RemoteModelError(
                "FIREWORKS_API_KEY is not set; cannot call the remote model."
            )

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = self._post_with_retry(url, payload, headers)

        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RemoteModelError(f"Unexpected Fireworks response shape: {data}") from exc

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        return RemoteResult(
            text=text,
            tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
