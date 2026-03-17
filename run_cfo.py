# pylint: disable=missing-module-docstring
# pylint: disable=no-name-in-module
"""
CLI entry point for CFO with FlowMol + Adjoint Matching.

Wires together:
  - Model: FlowMol (pretrained flow-based generative model)
  - FineTuningSolver: Adjoint Matching
  - Algorithm: CFO (Constrained Flow Optimization)
"""
import sys
from pathlib import Path

# Add src/ to Python path so that cfo and finetuning_solver are importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import os.path as osp
import time
import copy
import wandb
import pandas as pd
import torch
import torch.nn as nn
import dgl
from datetime import datetime
from omegaconf import OmegaConf

import flowmol
from cfo import AugmentedLagrangian, AugmentedReward, run_cfo, set_seed
from finetuning_solver.adjoint_matching import AdjointMatchingFinetuningTrainerFlowMol
from regressor import GNN, EGNN
from true_rc import pred_vs_real
from utils import parse_arguments, update_config_with_args
from utils.sampling import sampling


def setup_gen_model(flow_model: str, device: torch.device):
    gen_model = flowmol.load_pretrained(flow_model)
    gen_model.to(device)
    return gen_model


def load_regressor(config: OmegaConf, device: torch.device) -> nn.Module:
    K_x = 3   # spatial dimensions
    K_a = 10  # atom features
    K_c = 6   # charge classes
    K_e = 5   # bond types
    print(config.fn, config.model_type, config.date)
    model_path = osp.join("pretrained_models", str(config.fn), str(config.model_type), str(config.date), "best_model.pt")
    state = torch.load(model_path, map_location=device)
    if config.model_type == "gnn":
        model_config = {
            "property": state["config"]["property"],
            "node_feats": K_a + K_c + K_x,
            "edge_feats": K_e,
            "hidden_dim": state["config"]["hidden_dim"],
            "depth": state["config"]["depth"],
        }
        model = GNN(**model_config)
    elif config.model_type == "egnn":
        model_config = {
            "property": state["config"]["property"],
            "num_atom_types": K_a,
            "num_charge_classes": K_c,
            "num_bond_types": K_e,
            "hidden_dim": state["config"]["hidden_dim"],
            "depth": state["config"]["depth"],
        }
        model = EGNN(**model_config)
    model.load_state_dict(state["model_state"])
    return model


def main():
    args = parse_arguments()

    # Load config
    config_path = Path("configs/cfo.yaml")
    config = OmegaConf.load(config_path)
    config = update_config_with_args(config, args)
    baseline = args.baseline
    config.baseline = baseline

    # Seed and device
    set_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # WandB
    use_wandb = args.use_wandb and not args.debug
    run_id = datetime.now().strftime("%m%d_%H%M")
    if config.experiment is not None and "sweep" in config.experiment:
        run_name = f"lu{config.augmented_lagrangian.lagrangian_updates}_rl{config.reward_lambda}_rho{config.augmented_lagrangian.rho_init}_seed{config.seed}"
    elif config.experiment is not None:
        run_name = f"{run_id}_{config.experiment}_{config.seed}"
    else:
        run_name = f"{run_id}_r{config.reward.model_type}_c{config.constraint.model_type}{config.constraint.bound}_rf{config.reward_lambda}_lu{config.augmented_lagrangian.lagrangian_updates}"
    print(f"Running: {run_name}")
    if use_wandb:
        wandb.init(name=run_name, config=OmegaConf.to_container(config, resolve=True))
        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else None
        if sweep_id is not None:
            print(f"WandB sweep ID: {sweep_id} - Run ID: {wandb.run.id}", flush=True)

    # Save path
    save_path = Path(config.root) / Path("aa_experiments")
    if use_wandb and sweep_id is not None:
        save_path = save_path / Path(f"{config.experiment}") / Path(f"{config.seed}_{wandb.run.id}")
    else:
        save_path = save_path / Path(f"{config.experiment}") / Path(f"{run_name}")
    if (args.save_samples or args.save_model or args.save_plots) and not args.debug and not use_wandb:
        save_path = save_path / Path(run_name)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Run will be saved at:")
        print(save_path)

    # Molecular generation parameters
    n_atoms = config.get("n_atoms", None)
    min_num_atoms = config.get("min_num_atoms", None)
    max_num_atoms = config.get("max_num_atoms", None)

    config.adjoint_matching.sampling.n_atoms = n_atoms
    config.adjoint_matching.sampling.min_num_atoms = min_num_atoms
    config.adjoint_matching.sampling.max_num_atoms = max_num_atoms

    # Parameters
    lagrangian_updates = config.augmented_lagrangian.lagrangian_updates
    reward_lambda = config.reward_lambda
    config.adjoint_matching.reward_lambda = reward_lambda
    num_iterations = config.total_steps // lagrangian_updates
    plotting_freq = 3 if args.plotting_freq is None else args.plotting_freq

    # Load models
    base_model = setup_gen_model(config.flow_model, device=device)
    fine_model = copy.deepcopy(base_model)
    reward_model = load_regressor(config.reward, device=device)
    constraint_model = load_regressor(config.constraint, device=device)

    # Debug overrides
    if args.debug:
        config.augmented_lagrangian.sampling.num_samples = 8
        config.adjoint_matching.sampling.num_samples = 20 if torch.cuda.is_available() else 4
        config.adjoint_matching.batch_size = 5 if torch.cuda.is_available() else 2
        config.reward_sampling.num_samples = 8
        plotting_freq = 1
        args.save_samples = False
        num_iterations = 2
        lagrangian_updates = 2
        print("Debug mode activated", flush=True)

    # Print config
    print(f"--- Start ---", flush=True)
    print(f"Finetuning {config.flow_model}", flush=True)
    print(f"Reward: {config.reward.fn} - Constraint: {config.constraint.fn}", flush=True)
    print(f"Maximum Bound: {config.constraint.bound}", flush=True)
    print(f"Device: {device}", flush=True)
    start_time = time.time()

    # --- Wire up CFO components ---

    # 1. AugmentedReward (computes f_k in Algorithm 1)
    augmented_reward = AugmentedReward(
        reward_fn=reward_model,
        constraint_fn=constraint_model,
        alpha=reward_lambda,
        bound=config.constraint.bound,
        device=device,
        baseline=baseline,
    )
    augmented_reward.set_lambda_rho(
        lambda_=config.augmented_lagrangian.lambda_init,
        rho_=0.0 if baseline else config.augmented_lagrangian.rho_init,
    )

    # 2. AugmentedLagrangian (manages λ, ρ updates)
    alm = AugmentedLagrangian(
        config=config.augmented_lagrangian,
        constraint_fn=constraint_model,
        bound=config.constraint.bound,
        device=device,
        baseline=baseline,
    )

    # 3. FineTuningSolver factory (creates AM trainer each ALM round)
    def create_trainer_fn(config, model, base_model, grad_reward_fn, device, verbose):
        return AdjointMatchingFinetuningTrainerFlowMol(
            config=config,
            model=model,
            base_model=base_model,
            grad_reward_fn=grad_reward_fn,
            device=device,
            verbose=verbose,
        )

    # 4. Sampling function (FlowMol-specific)
    def sampling_fn(model):
        return sampling(
            config.reward_sampling,
            model,
            device=device,
            n_atoms=n_atoms,
            min_num_atoms=min_num_atoms,
            max_num_atoms=max_num_atoms,
        )

    # 5. Optional sample saving
    save_samples_fn = None
    if args.save_samples and not args.debug:
        from dgl import save_graphs
        sample_path = save_path / Path("samples")
        sample_path.mkdir(parents=True, exist_ok=True)
        def save_samples_fn(dgl_mols, step_idx):
            save_graphs(str(sample_path / Path(f"samples_{step_idx}.bin")), dgl_mols)

    # 6. Evaluation function (pred vs real, optional)
    def eval_fn(rd_mols, dgl_mols_batched, pred_rc):
        log_eval, _, _ = pred_vs_real(rd_mols, dgl_mols_batched, pred_rc,
                                       reward=config.reward.fn, constraint=config.constraint.fn)
        return log_eval

    # --- Run CFO ---
    fine_model, full_stats, al_stats, models_list = run_cfo(
        config=config,
        base_model=base_model,
        fine_model=fine_model,
        augmented_reward=augmented_reward,
        alm=alm,
        create_trainer_fn=create_trainer_fn,
        sampling_fn=sampling_fn,
        device=device,
        num_iterations=num_iterations,
        lagrangian_updates=lagrangian_updates,
        plotting_freq=plotting_freq,
        use_wandb=use_wandb,
        verbose=args.debug,
        save_samples_fn=save_samples_fn,
        eval_fn=eval_fn,
    )

    # --- Post-processing: save stats, models, plots ---
    for feat in ['loss', 'grad_norm', 'adj_0_norm']:
        full_stats[0][feat] = full_stats[1][feat]
    df_al = pd.DataFrame.from_records(full_stats)
    df_alm = pd.DataFrame.from_dict(al_stats)

    if not args.debug:
        save_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, save_path / Path("config.yaml"))
        df_al.to_csv(save_path / "full_stats.csv", index=False)
        df_alm.to_csv(save_path / "al_stats.csv", index=False)
        print(f"Saved stats to {save_path}", flush=True)

    if args.save_model and not args.debug:
        for idx, state_dict in enumerate(models_list):
            torch.save(state_dict, save_path / Path(f"model_lu{idx+1}.pth"))
        torch.save(fine_model.cpu().state_dict(), save_path / Path("final_model.pth"))
        print(f"Model saved to {save_path}", flush=True)

    print(f"--- Final ---", flush=True)
    print(f"Final reward: {full_stats[-1]['reward']:.4f}", flush=True)
    print(f"Duration: {(time.time()-start_time)/60:.2f} mins", flush=True)
    print()


if __name__ == "__main__":
    main()
