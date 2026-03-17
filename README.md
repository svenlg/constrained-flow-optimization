# Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design

[![Paper](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/scholar?q=Constrained+Flow+Optimization+via+Sequential+Fine-Tuning+for+Molecular+Design)

---

## Overview

Given a pretrained generative flow model $p^\pi_\text{pre}$ and differentiable reward/constraint predictors, CFO solves:

$$\max_\pi \mathbb{E}_{x \sim p^\pi_1}[r(x)] - \alpha \, D_\text{KL}(p^\pi_1 \| p^\text{pre}_1) \quad \text{s.t.} \quad \mathbb{E}_{x \sim p^\pi_1}[c(x)] \leq B$$

by alternating between:
1. **Fine-tuning** the flow model via KL-regularized Adjoint Matching (Algorithm 2)
2. **Updating** the Lagrange multiplier $\lambda$ and penalty $\rho$ via an Augmented Lagrangian Method (Algorithm 1, Steps 3-5)

The framework is **model-agnostic** — only the fine-tuning solver is model-specific. We provide an implementation for [FlowMol](https://github.com/Dunni3/FlowMol) (a flow matching model for 3D molecular generation).

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
# Debug run (small samples, CPU-compatible)
python run_cfo.py --debug

# Full run with logging
python run_cfo.py --use_wandb --experiment "dipole_energy" --seed 42

# Custom reward/constraint
python run_cfo.py --reward dipole --constraint energy --bound -80 --reward_lambda 50
```

## Configuration

Configs live in `configs/` (gitignored). Create `configs/cfo.yaml`:

CLI flags override config values (e.g., `--reward_lambda 100 --bound -90`).

### CLI Flags

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

## Citation

```bibtex
@inproceedings{gutjahr2025constrained,
  title={Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design},
  author={Gutjahr, Sven and De Santi, Riccardo and Schaufelberger, Luca and Jorner, Kjell and Krause, Andreas},
  booktitle={NeurIPS 2025 Workshop on Structured Probabilistic Inference & Generative Modeling}
}
```

## License

MIT
