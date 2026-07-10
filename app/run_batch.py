"""Track 1 batch submission entry point.

Reads a JSON array of tasks from ``INPUT_PATH`` (default ``/input/tasks.json``),
routes each through the **existing** ``Agent`` (same wiring as the server — no
routing/executor logic is duplicated or changed here), writes a JSON array of
answers to ``OUTPUT_PATH`` (default ``/output/results.json``), and exits 0.

Robust by contract:
* a missing / empty / non-array / malformed input writes ``[]`` and exits 0;
* a per-task routing error writes an empty answer for that task, never aborts;
* the id / prompt / answer field names are env-overridable, so a schema mismatch
  is fixed with an env var instead of a rebuild — and the field names actually
  used are echoed to stdout for verification against the real sample.

Env knobs (all optional):
    INPUT_PATH           default /input/tasks.json
    OUTPUT_PATH          default /output/results.json
    TASK_ID_FIELD        default "id"          (fallback: array index)
    TASK_PROMPT_FIELD    default "prompt,query" (first present wins)
    RESULT_ANSWER_FIELD  default "answer"
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Tuple

INPUT_PATH = os.getenv("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")
ID_FIELD = os.getenv("TASK_ID_FIELD", "id")
PROMPT_FIELDS = [f.strip() for f in os.getenv("TASK_PROMPT_FIELD", "prompt,query").split(",") if f.strip()]
ANSWER_FIELD = os.getenv("RESULT_ANSWER_FIELD", "answer")
# "objects" (default) -> [{"id":…, "answer":…}] ; "strings" -> ["…", …] in task order.
RESULT_FORMAT = os.getenv("RESULT_FORMAT", "objects").strip().lower()
_BARE = RESULT_FORMAT in ("strings", "string", "bare")


def _log(msg: str) -> None:
    print(f"[run_batch] {msg}", flush=True)


def _emit(tid, answer):
    """One result entry in the configured shape (object vs bare string)."""
    return answer if _BARE else {ID_FIELD: tid, ANSWER_FIELD: answer}


def _load_tasks(path: str) -> Optional[list]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:  # missing file or invalid JSON
        _log(f"cannot read {path!r}: {exc} -> writing empty results")
        return None
    if not isinstance(data, list):
        _log(f"{path!r} is not a JSON array (got {type(data).__name__}) -> writing empty results")
        return None
    return data


def _prompt_of(task) -> Tuple[Optional[str], str]:
    if isinstance(task, dict):
        for f in PROMPT_FIELDS:
            val = task.get(f)
            if val:
                return f, str(val)
    elif isinstance(task, str):  # tolerate a bare-string task
        return "(string)", task
    return None, ""


def _write(path: str, results: list) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)


def run() -> int:
    _log(f"input={INPUT_PATH!r} output={OUTPUT_PATH!r} "
         f"id_field={ID_FIELD!r} prompt_fields={PROMPT_FIELDS} answer_field={ANSWER_FIELD!r}")

    tasks = _load_tasks(INPUT_PATH)
    if not tasks:
        _write(OUTPUT_PATH, [])
        _log(f"no valid tasks -> wrote [] to {OUTPUT_PATH!r}; exit 0")
        return 0

    # Reuse the exact server wiring so routing/executor behaviour is identical.
    from app.main import get_agent

    try:
        agent = get_agent()
    except Exception as exc:  # noqa: BLE001 — never crash the submission
        _log(f"FATAL: could not build agent ({exc}); writing [] and exiting 0")
        _write(OUTPUT_PATH, [])
        return 0

    results: List[dict] = []
    used_prompt_field: Optional[str] = None
    for idx, task in enumerate(tasks):
        tid = task.get(ID_FIELD, idx) if isinstance(task, dict) else idx
        prompt_field, prompt = _prompt_of(task)
        if prompt_field and used_prompt_field is None:
            used_prompt_field = prompt_field
        if not prompt:
            _log(f"id={tid}: no prompt in fields {PROMPT_FIELDS} -> empty answer")
            results.append(_emit(tid, ""))
            continue
        try:
            r = agent.route(prompt)
            results.append(_emit(tid, r.answer))
            # Per-task audit line (NOT written to results.json) so you can verify
            # the executor fired at zero remote tokens on math/code tasks.
            _log(f"id={tid} route={r.route} remote_tokens={r.remote_tokens} local_tokens={r.local_tokens}")
        except Exception as exc:  # noqa: BLE001 — one bad task must not abort the run
            _log(f"id={tid} ERROR: {exc} -> empty answer")
            results.append(_emit(tid, ""))

    _write(OUTPUT_PATH, results)
    shape = "bare-strings" if _BARE else f"objects(id={ID_FIELD!r}, answer={ANSWER_FIELD!r})"
    _log(f"wrote {len(results)} results to {OUTPUT_PATH!r} | shape={shape} | "
         f"prompt_field_used={used_prompt_field!r}")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:  # noqa: BLE001 — last-resort guard: still exit 0 with []
        _log(f"UNEXPECTED: {exc}; writing [] and exiting 0")
        try:
            _write(OUTPUT_PATH, [])
        except Exception:  # noqa: BLE001
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
