# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Constrained Flow Optimization (CFO) framework for fine-tuning pretrained flow/diffusion models under constraints, accompanying the ICLR 2026 paper.

## Running Experiments

```bash
python run_cfo.py --debug                                    # Quick local test
python run_cfo.py --use_wandb --experiment "my_run" --seed 42  # Full run
```

**Key CLI flags** (override YAML config values):
- `--reward {dipole,score,energy,sascore}` / `--constraint {dipole,score,energy,sascore}`
- `--bound <float>` — constraint bound B
- `--reward_lambda <float>` — KL regularization strength alpha
- `--total_steps <int>` / `--lagrangian_updates <int>`
- `--baseline` — fixed lambda (no ALM updates)
- `--save_model` / `--save_samples`

## Architecture (three separate parts)

### `src/cfo/` — CFO Algorithm (Algorithm 1, model-agnostic)
- **`alm.py`** — `AugmentedLagrangian`: manages lambda/rho updates (Steps 3-5)
- **`augmented_reward.py`** — `AugmentedReward`: computes augmented objective f_k (Step 1)
- **`runner.py`** — `run_cfo()`: the ALM outer loop, accepts pluggable model + solver
- **`utils.py`** — `set_seed()`

### `src/finetuning_solver/adjoint_matching/` — FineTuningSolver (Algorithm 2, FlowMol-specific)
- **`loss.py`** — Generic AM loss functions: `adj_matching_loss`, `AMDataset`, `create_timestep_subset`
- **`trainer.py`** — `AdjointMatchingFinetuningTrainerFlowMol`: FlowMol-specific trainer with `generate_dataset()` and `finetune()`
- **`solver.py`** — `LeanAdjointSolverFlow`: adjoint ODE solver using FlowMol's vector field

### Other modules
- **`flowmol/`** — Pretrained FlowMol generative model. Loaded via `flowmol.load_pretrained("geom_gaussian")`.
- **`regressor/`** — GNN and EGNN property predictors. Weights in `pretrained_models/{fn}/{model_type}/best_model.pt`.
- **`utils/`** — CLI argument parsing (`setup.py`), FlowMol sampling helper (`sampling.py`), plotting utilities.
- **`auxiliary/`** — Regressor training, property calculation (not part of main pipeline).
- **`notebooks/`** — Demo notebooks (`cfo_quickstart.ipynb`, `cfo_step_by_step.ipynb`) + paper figure notebooks.

### How `run_cfo.py` wires components together
```python
run_cfo(
    config, base_model, fine_model,     # Model: FlowMol
    augmented_reward, alm,               # CFO: ALM + augmented objective
    create_trainer_fn,                   # Solver factory: -> AdjointMatchingTrainer
    sampling_fn,                         # Sampling: FlowMol-specific
    device, num_iterations, lagrangian_updates,
)
```

## Configuration

YAML configs in `configs/` (tracked). Config loaded with OmegaConf, CLI args override via `utils/setup.py:update_config_with_args()`.

**Import note:** `run_cfo.py` adds `src/` to `sys.path` so imports are `from cfo import ...` and `from finetuning_solver.adjoint_matching import ...`.

## Gotchas

- **Pretrained model paths** — regressor weights at `pretrained_models/{fn}/{model_type}/best_model.pt`. Only `dipole/egnn` and `energy/gnn` are included.
- **`--debug` changes sample counts** — overrides num_samples, batch_size. Don't compare debug runs to production runs.
- **`src/` path setup** — entry points must add `src/` to `sys.path` before importing `cfo` or `finetuning_solver`.
