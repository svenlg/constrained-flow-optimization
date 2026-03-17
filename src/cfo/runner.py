"""
CFO Algorithm 1: Constrained Flow Optimization.

Implements the augmented Lagrangian outer loop from the paper.
Model-agnostic and solver-agnostic — accepts pluggable components.
"""
import time
import copy
import torch
import dgl
import wandb


def run_cfo(
    config,
    base_model,
    fine_model,
    augmented_reward,
    alm,
    create_trainer_fn,
    sampling_fn,
    device,
    num_iterations,
    lagrangian_updates,
    plotting_freq=3,
    use_wandb=False,
    verbose=False,
    save_samples_fn=None,
    eval_fn=None,
):
    """
    CFO Algorithm 1: Constrained Flow Optimization via Sequential Fine-Tuning.

    Alternates between:
      - Step 2: Solving a KL-regularized fine-tuning subproblem via create_trainer_fn
      - Steps 3-5: Updating λ, ρ via alm.update_lambda_rho()

    Args:
        config: OmegaConf config object
        base_model: π_pre — the pretrained generative model
        fine_model: Copy of base_model to be fine-tuned (will be modified in-place)
        augmented_reward: AugmentedReward instance (wraps r, c, B, computes f_k)
        alm: AugmentedLagrangian instance (manages λ, ρ updates)
        create_trainer_fn: Factory callable(config, model, base_model, grad_fn, device, verbose)
            -> trainer with .generate_dataset() and .finetune() methods
        sampling_fn: callable(config, model, device, ...) -> (dgl_mols_list, rd_mols_list)
        device: torch.device
        num_iterations: N — number of FineTuningSolver iterations per ALM round
        lagrangian_updates: K — number of ALM outer rounds
        plotting_freq: how often to evaluate (every N steps)
        use_wandb: whether to log to W&B
        verbose: print debug info
        save_samples_fn: optional callable(dgl_mols, step_idx) to save samples
        eval_fn: optional callable(rd_mols, dgl_mols_batched, pred_rc) -> dict
            Evaluates predicted vs real properties. Result dict is merged into stats.

    Returns:
        fine_model: the final fine-tuned model
        full_stats: list of dicts with per-step statistics
        al_stats: list of dicts with per-ALM-round statistics
        models_list: list of state_dicts (one per ALM round)
    """
    al_stats = []
    full_stats = []
    models_list = []
    al_best_epoch = 0

    # --- Initial evaluation ---
    tmp_model = copy.deepcopy(fine_model)
    dgl_mols, rd_mols = sampling_fn(tmp_model)
    del tmp_model

    if save_samples_fn is not None:
        save_samples_fn(dgl_mols, 0)

    # Compute reward for initial samples
    dgl_mols_batched = dgl.batch(dgl_mols).to(device)
    _ = augmented_reward(dgl_mols_batched)
    tmp_log = augmented_reward.get_statistics()
    full_stats.append(tmp_log)

    # Evaluate predicted vs real properties
    if eval_fn is not None:
        pred_rc = augmented_reward.get_reward_constraint()
        log_eval = eval_fn(rd_mols, dgl_mols_batched, pred_rc)
        full_stats[-1].update(log_eval)
    del dgl_mols_batched

    al_lowest_const = full_stats[-1]["constraint"]
    al_best_reward = full_stats[-1]["reward"]

    # Set initial expected constraint (for logging)
    _ = alm.expected_constraint(dgl_mols)

    # Log initial state
    if use_wandb:
        logs = {}
        logs.update(tmp_log)
        if eval_fn is not None:
            logs.update(log_eval)
        log = alm.get_statistics()
        logs.update(log)
        wandb.log(logs)

    reward_lambda = config.reward_lambda

    # --- AUGMENTED LAGRANGIAN OUTER LOOP (Algorithm 1, for k=1..K) ---
    alg_time = time.time()
    total_steps_made = 0

    for k in range(1, lagrangian_updates + 1):
        print(f"--- AL Round {k}/{lagrangian_updates} ---", flush=True)

        # Step 3-5 from previous round: get current λ, ρ
        lambda_, rho_ = alm.get_current_lambda_rho()
        log = alm.get_statistics()
        al_stats.append(log)

        # Step 1: Update augmented objective f_k
        augmented_reward.set_lambda_rho(lambda_, rho_)
        print(f"Lambda: {lambda_:.4f}, rho: {rho_:.4f}", flush=True)

        # Step 2: Solve fine-tuning subproblem via FineTuningSolver
        trainer = create_trainer_fn(
            config=config.adjoint_matching,
            model=copy.deepcopy(fine_model),
            base_model=copy.deepcopy(base_model),
            grad_reward_fn=augmented_reward.grad_augmented_reward_fn,
            device=device,
            verbose=verbose,
        )

        am_stats = []
        am_best_total_reward = -1e8
        am_best_iteration = 0

        # Inner loop: N iterations of the FineTuningSolver
        for i in range(1, num_iterations + 1):
            dataset, avg_adj_0_norm = trainer.generate_dataset()

            if dataset is None:
                print("Dataset is None, skipping iteration", flush=True)
                continue

            loss, grad_norm = trainer.finetune(dataset)
            del dataset

            total_steps_made += 1
            if total_steps_made % plotting_freq == 0:
                # Evaluate current model
                tmp_model = copy.deepcopy(trainer.fine_model)
                dgl_mols, rd_mols = sampling_fn(tmp_model)
                del tmp_model

                if save_samples_fn is not None:
                    save_samples_fn(dgl_mols, total_steps_made)

                dgl_mols_batched = dgl.batch(dgl_mols).to(device)
                _ = augmented_reward(dgl_mols_batched)
                pred_rc = augmented_reward.get_reward_constraint()
                if eval_fn is not None:
                    log_eval = eval_fn(rd_mols, dgl_mols_batched, pred_rc)
                tmp_log = augmented_reward.get_statistics()
                tmp_log["adj_0_norm"] = avg_adj_0_norm
                tmp_log["loss"] = loss
                tmp_log["loss_re_weighted"] = loss / (reward_lambda ** 2)
                tmp_log["grad_norm"] = grad_norm
                if eval_fn is not None:
                    tmp_log.update(log_eval)
                am_stats.append(tmp_log)

                if am_stats[-1]["total_reward"] > am_best_total_reward:
                    am_best_total_reward = am_stats[-1]["total_reward"]
                    am_best_iteration = i

                if use_wandb:
                    logs = {}
                    logs.update(tmp_log)
                    if eval_fn is not None:
                        logs.update(log_eval)
                    log = alm.get_statistics()
                    logs.update(log)
                    wandb.log(logs)

                del dgl_mols, dgl_mols_batched, rd_mols, pred_rc, tmp_log

                print(f"\tIteration {i}: Total Reward: {am_stats[-1]['total_reward']:.4f}, "
                      f"Reward: {am_stats[-1]['reward']:.4f}, "
                      f"Constraint: {am_stats[-1]['constraint']:.4f}, "
                      f"Violations: {am_stats[-1]['constraint_violations']:.4f}", flush=True)
                print(f"\tBest reward: {am_best_total_reward:.4f} in step {am_best_iteration}", flush=True)

        full_stats.extend(am_stats)
        fine_model = copy.deepcopy(trainer.fine_model)

        # Track best
        if full_stats[-1]["constraint"] < al_lowest_const:
            al_lowest_const = full_stats[-1]["constraint"]
            al_best_epoch = k
            al_best_reward = full_stats[-1]["reward"]

        print(f"Best overall reward: {al_best_reward:.4f} with violations {al_lowest_const:.4f} at epoch {al_best_epoch}", flush=True)

        # Step 3-5: Sample + update λ, ρ
        tmp_model = copy.deepcopy(fine_model)
        dgl_mols, rd_mols = sampling_fn(tmp_model)
        del tmp_model

        # Save model checkpoint
        models_list.append(copy.deepcopy(fine_model.cpu().state_dict()))
        fine_model.to(device)

        alm.update_lambda_rho(dgl_mols)
        del dgl_mols, rd_mols, trainer

    alg_time = time.time() - alg_time
    print()
    print(f"--- Finished --- {total_steps_made} total-steps ---", flush=True)
    print(f"Time: {alg_time/60:.2f} mins", flush=True)
    print()

    if use_wandb:
        wandb.finish()

    return fine_model, full_stats, al_stats, models_list
