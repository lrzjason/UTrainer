# UTrainer

A modular, config-driven diffusion model training framework (**UnifiedTrainer**) plus a
multi-GPU training orchestrator with SQLite-backed job scheduling, CLI, HTTP API and a
built-in web frontend.

- **UnifiedTrainer/** — protocol-based training framework: one entry point (`train.py`),
  config-driven assembly, plug-and-play model adapters and composable losses.
- **orchestrator/** — watcher → scheduler → dispatcher pipeline that runs `train.py`
  as managed worker subprocesses (inbox files, hooks, suspend/resume, GPU guard).
- **agent/** · **md/** · **progress/** — design docs, decision log and implementation
  history (see [Agent docs](#agent-docs)).

> Note: design/implementation docs under `md/` are written in Chinese.

## Repository layout

```
UTrainer/
├── UnifiedTrainer/        # Training framework (train.py + models/losses/data/engine)
├── orchestrator/          # Job orchestration: cli, db, scheduler, dispatcher, server
├── web/dist/              # Prebuilt web frontend (served by the orchestrator API)
├── workspace/             # Runtime state: inbox/processing/done/failed + trainer.db
├── md/                    # Feature design docs (00-architecture.md is the index)
├── agent/                 # decisions.md — design decision log
├── progress/              # Implementation logs (001–011)
├── skills/                # Agent skill: unified-trainer-orchestrator
├── doc/                   # Overall design (Improvement_k3.md), REQUIREMENTS.md
├── start.bat / start.sh   # Windows / Linux orchestrator start scripts
└── requirements.txt
```

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the orchestrator (API + web UI on http://127.0.0.1:7860, 2 parallel workers)
# Windows: double-click start.bat, or:
python -m orchestrator.main --workspace workspace --api --port 7860 --max-parallel 2
```

## Using UnifiedTrainer (training framework)

Train a model directly (without the orchestrator):

```bash
cd UnifiedTrainer

# List available models / losses
python train.py --model dummy --config dummy --list-models
python train.py --model dummy --config dummy --list-losses

# Train (configs live in UnifiedTrainer/configs/)
python train.py --model flux2_klein --config configs/flux2_klein_example.json
python train.py --model krea2      --config configs/krea2_example.json

# Resume from a checkpoint
python train.py --model flux2_klein --config configs/flux2_klein_example.json \
    --resume output/checkpoint.safetensors
```

**Configs**: real training configs are gitignored (keep local/private). Committed
examples follow the `*_example.json` naming under `UnifiedTrainer/configs/`
(e.g. `flux2_klein_example.json`, `krea2_example.json`). Copy an example, fill in
your model/data paths, and run.

**Config format** (see AGENTS.md / example files for details):

```json
{
    "model": "krea2",
    "model_path": "/path/to/Krea-2-Raw",
    "data": {
        "cache_dir": "cache/my_cache",
        "dataset_configs": [{"train_data_dir": "/path/to/images", "resolution": 1024}],
        "num_workers": 4
    },
    "losses": [{"type": "flow_matching", "weight": 1.0}],
    "training": {
        "batch_size": 1, "learning_rate": 1e-4, "num_epochs": 20,
        "network_type": "lokr", "lora_rank": 64, "quantize": "torchao_float8",
        "weight_dtype": "bf16", "gradient_checkpointing": true
    },
    "output": {"dir": "output/experiment", "save_name": "my_lora", "checkpoint_every": 500}
}
```

**Supported models**: `flux2_klein`, `qwen_image`, `qwen_image_edit`, `z_image`,
`sd3`, `sd35`, `krea2`, `hidream`.
**Supported losses**: `flow_matching`, `lcs`, `lcs_saturation`, `mlp_lab`, `l2_reg`.

## Using the Orchestrator

The orchestrator manages training tasks: submit JSON job files into `workspace/inbox/`,
or use the CLI / HTTP API / web UI. Tasks run as `train.py` subprocesses with
heartbeat reporting, hook support (sample / save / restore / patch_config / suspend)
and GPU VRAM admission control.

```bash
# Main service (watcher + scheduler + dispatcher + API)
python -m orchestrator.main --workspace workspace --api --port 7860 --max-parallel 2

# CLI examples
python -m orchestrator.cli project create demo --model krea2
python -m orchestrator.cli submit job.json                     # copy into inbox/
python -m orchestrator.cli task create --project demo --name exp1 --config config.json
python -m orchestrator.cli list --project demo
python -m orchestrator.cli hook 1 sample --n 4
python -m orchestrator.cli hook 1 suspend                      # then: resume 1
python -m orchestrator.cli gpu status
```

Full CLI reference, HTTP API routes and the task state machine:
[`md/ORCHESTRATOR.md`](md/ORCHESTRATOR.md).

## Agent docs

Where to find documentation for AI agents working on this repo:

| Path | Content |
|------|---------|
| [`md/00-architecture.md`](md/00-architecture.md) | **Doc index** — overall architecture + feature doc table |
| [`md/01-database.md`](md/01-database.md) … [`md/06-cli-scheduler-multigpu.md`](md/06-cli-scheduler-multigpu.md) | Per-feature design docs (DB, orchestrator core, hooks, GPU scheduler, API/frontend, CLI/multi-GPU) |
| [`md/ORCHESTRATOR.md`](md/ORCHESTRATOR.md) | Orchestrator usage manual (as-built) |
| [`agent/decisions.md`](agent/decisions.md) | Design decision log (what was decided and why) |
| [`progress/`](progress/) | Implementation logs `001`–`011` (P1–P5 phases, incl. code reviews) |
| [`doc/Improvement_k3.md`](doc/Improvement_k3.md) | Overall design document |
| [`doc/REQUIREMENTS.md`](doc/REQUIREMENTS.md) | Project requirements (no virtualenv policy, etc.) |
| [`skills/unified-trainer-orchestrator/SKILL.md`](skills/unified-trainer-orchestrator/SKILL.md) | Agent skill for orchestrator workflows |
| [`../AGENTS.md`](../AGENTS.md) | Quickstart for AI agents (parent workspace, outside this repo) |

## Key architectural decisions

- SQLite (`workspace/trainer.db`) is the **single source of truth**; JSON files are
  import-only and never written back.
- Orchestrator and workers (train.py subprocesses) are separated; workers communicate
  via exit codes (0=done, 42=suspend, other=failed).
- Frontend, CLI and inbox files are three equivalent entry paths sharing the same
  `db.py`/`hooks.py` functions.
- Hierarchy: Project → Task (chained/scheduled/standalone) → Hook commands.
