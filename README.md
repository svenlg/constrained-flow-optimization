# Constrained Flow Optimization (CFO)

Official code for **"Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design"** (ICLR 2026).

> CFO fine-tunes a pretrained flow model to maximize a reward (e.g., dipole moment) while satisfying constraints (e.g., energy bound), using an augmented Lagrangian outer loop with Adjoint Matching as the inner solver.

<!-- [![Paper](https://img.shields.io/badge/arXiv-...)](https://arxiv.org/abs/...) -->

---

## Overview

Given a pretrained generative flow model $\pi_\text{pre}$ and differentiable reward/constraint predictors, CFO solves:

$$\max_\theta \; \mathbb{E}_{x \sim \pi_\theta}[r(x)] \quad \text{s.t.} \quad \mathbb{E}_{x \sim \pi_\theta}[c(x)] \leq B$$

by alternating between:
1. **Fine-tuning** the flow model via KL-regularized Adjoint Matching (Algorithm 2)
2. **Updating** the Lagrange multiplier $\lambda$ and penalty $\rho$ via an Augmented Lagrangian Method (Algorithm 1, Steps 3-5)

The framework is **model-agnostic** — only the fine-tuning solver is model-specific. We provide an implementation for [FlowMol](https://github.com/Dunni3/FlowMol) (a flow matching model for 3D molecular generation).

## Repository Structure

```
run_cfo.py                        # Main entry point

src/
  cfo/                            # Algorithm 1: CFO (model-agnostic)
    runner.py                     #   Outer ALM loop
    alm.py                        #   Augmented Lagrangian (lambda/rho updates)
    augmented_reward.py           #   Augmented objective f_k(x)
    utils.py                      #   set_seed()

  finetuning_solver/              # Algorithm 2: FineTuningSolver
    adjoint_matching/             #   Adjoint Matching (Domingo-Enrich et al., 2025)
      solver.py                   #     Adjoint ODE solver
      loss.py                     #     AM loss (model-agnostic)
      trainer.py                  #     FlowMol-specific trainer

flowmol/                          # Pretrained FlowMol (included as subpackage)
regressor/                        # GNN and EGNN property predictors
pretrained_models/                # Pretrained regressor weights (not tracked)
configs/                          # YAML configs (not tracked, see below)
notebooks/                        # Demos and paper figure notebooks
utils/                            # CLI argument parsing, sampling, plotting
auxiliary/                        # Regressor training pipeline
true_rc/                          # Ground-truth property calculation (RDKit, xTB)
```

## Installation

```bash
git clone https://github.com/<user>/constrained-flow-optimization.git
cd constrained-flow-optimization

# Create conda environment
conda create -n cfo python=3.9
conda activate cfo
pip install torch dgl rdkit-pypi omegaconf wandb pandas
pip install -e .  # or: pip install -e flowmol/
```

### Pretrained Models

FlowMol weights are included under `flowmol/trained_models/`. Regressor weights must be placed under `pretrained_models/{property}/{model_type}/{date}/best_model.pt`.

Available properties: `dipole`, `energy`, `score`, `sascore`.
Model types: `gnn`, `egnn`.

## Quick Start

```bash
# Debug run (small samples, CPU-compatible, ~6 min)
python run_cfo.py --debug

# Full run with logging
python run_cfo.py --use_wandb --experiment "dipole_energy" --seed 42

# Custom reward/constraint
python run_cfo.py --reward dipole --constraint energy --bound -80 --reward_lambda 50
```

## Configuration

Configs live in `configs/` (gitignored). Create `configs/cfo.yaml`:

```yaml
seed: 42
experiment: "my_run"
flow_model: "geom_gaussian"      # geom_gaussian | geom_ctmc

# Reward function
reward:
  fn: dipole                     # dipole | energy | score | sascore
  model_type: egnn               # gnn | egnn
  date: "0922_1829"              # pretrained model date

# Constraint function
constraint:
  fn: energy
  bound: -80                     # upper bound B
  model_type: gnn
  date: "0923_1520"

# Augmented Lagrangian
reward_lambda: 50.0              # KL regularization alpha
total_steps: 60                  # total fine-tuning iterations
augmented_lagrangian:
  lagrangian_updates: 6          # ALM rounds K
  rho_init: 1.0
  eta: 1.25
  tau: 0.99

# Adjoint Matching
adjoint_matching:
  batch_size: 4
  lr: 5e-6
  clip_grad_norm: 1.0            # gradient clipping (recommended)
  cutoff_time: 0.5
  sampling:
    sampler_type: "memoryless"
    num_samples: 20
    num_integration_steps: 40
```

CLI flags override config values (e.g., `--reward_lambda 100 --bound -90`).

## CLI Flags

| Flag | Description |
|------|-------------|
| `--debug` | Small sample counts, fast iteration, no saving |
| `--reward {dipole,score,energy,sascore}` | Reward function |
| `--constraint {dipole,score,energy,sascore}` | Constraint function |
| `--bound <float>` | Constraint upper bound B |
| `--reward_lambda <float>` | KL regularization strength $\alpha$ |
| `--total_steps <int>` | Total fine-tuning iterations |
| `--lagrangian_updates <int>` | ALM outer loop rounds K |
| `--baseline` | Fixed $\lambda$ (no ALM updates) |
| `--save_model` / `--save_samples` | Save checkpoints and generated molecules |
| `--use_wandb` | Log to Weights & Biases |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `cfo_quickstart.ipynb` | Run CFO in a few lines |
| `cfo_step_by_step.ipynb` | Walk through Algorithm 1 step by step |
| `adjoint_matching.ipynb` | Adjoint Matching methodology |

## Adapting CFO to Another Model

CFO is model-agnostic. To use it with a different generative model:

1. **Keep** `src/cfo/` unchanged (ALM, AugmentedReward, runner)
2. **Implement** a new `FineTuningSolver` with:
   - `generate_dataset()` — sample trajectories and solve the adjoint ODE
   - `finetune(dataset)` — compute the AM loss and update model parameters
   - `fine_model` attribute — the model being fine-tuned
3. **Provide** a `sampling_fn(model) -> (dgl_mols, rdkit_mols)` for evaluation
4. **Provide** a `create_trainer_fn(config, model, base_model, grad_fn, device, verbose) -> trainer`

See `run_cfo.py` lines 184-237 for how the components are wired together.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  Algorithm 1: CFO (Augmented Lagrangian Loop)       │
│                                                     │
│  for k = 1, ..., K:                                 │
│    1. Update augmented objective f_k(λ_k, ρ_k)      │
│    2. Solve fine-tuning subproblem (Algorithm 2)    │
│    3. Sample from π_θ, evaluate constraint          │
│    4. Update λ_{k+1} ← max(λ_k + ρ_k·g_k, λ_min)    │
│    5. Update ρ_{k+1} (increase if needed)           │
└─────────────────────────────────────────────────────┘

```

## Citation

```bibtex
@inproceedings{
  gutjahr2026cfo,
  title={Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design},
  author={...},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={...}
}
```

## License

MIT
