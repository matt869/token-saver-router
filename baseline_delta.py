"""Baseline (naive, every task -> remote raw) vs TokenSaver (routing+IDS+compression).

Remote token counts are keyless-stub-derived (no Fireworks key on this box):
both columns use the SAME StubRemoteModel, so the comparison is apples-to-apples
on the remote side. Baseline sends the RAW prompt straight to remote with no
routing; TokenSaver runs the full agent.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import load_config
from app.models.local_model import LocalModel
from app.models.stub_remote import StubRemoteModel
from app.router.agent import Agent
from app.router.classifier import HeuristicClassifier

TASKS = json.loads(Path("app/eval/sample_tasks.json").read_text(encoding="utf-8"))
cfg = load_config()

# (a) NAIVE BASELINE: no routing, no compression — every prompt hits remote raw.
baseline_remote = StubRemoteModel()

# (b) TOKENSAVER: full routing + IDS + preflight compression, real local model.
ts_local = LocalModel(cfg.local_model, device=cfg.device, max_new_tokens=cfg.max_new_tokens)
ts_agent = Agent(
    config=cfg,
    classifier=HeuristicClassifier(length_cutoff=cfg.complexity_length_cutoff),
    local_model=ts_local,
    remote_model=StubRemoteModel(),
)

print("=" * 82)
print("BASELINE (naive: all->remote, raw)  vs  TOKENSAVER (routing+IDS+compression)")
print(f"remote token source: keyless StubRemoteModel (stub-derived, both columns)")
print(f"tokensaver local model: {cfg.local_model} @ max_new_tokens={cfg.max_new_tokens}")
print("=" * 82)
print(f"{'id':>2}  {'baseline_remote':>15}  {'tokensaver_remote':>17}  {'ts_route':<26}")
print("-" * 82)

base_tot = ts_tot = 0
for t in TASKS:
    q = t["query"]
    base_tok = baseline_remote.generate(q).tokens          # raw prompt -> remote
    r = ts_agent.route(q)
    base_tot += base_tok
    ts_tot += r.remote_tokens
    print(f"{t['id']:>2}  {base_tok:>15}  {r.remote_tokens:>17}  {r.route:<26}")

print("-" * 82)
print(f"{'':>2}  {base_tot:>15}  {ts_tot:>17}  {'<- TOTAL remote tokens':<26}")
pct = 100.0 * (base_tot - ts_tot) / base_tot if base_tot else 0.0
print("=" * 82)
print(f"REMOTE-TOKEN REDUCTION: {base_tot} -> {ts_tot}  =  {pct:.1f}% fewer scored tokens")
print("=" * 82)
