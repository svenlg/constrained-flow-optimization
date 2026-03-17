"""
Adjoint Matching Fine-Tuning Trainer for FlowMol.

This is the FlowMol-specific implementation of the FineTuningSolver (Algorithm 2 in paper).
To adapt CFO to another model, create a new trainer with the same interface:
  - __init__(config, model, base_model, grad_reward_fn, device, verbose)
  - generate_dataset() -> (dataset, avg_adj_0_norm)
  - finetune(dataset) -> (loss, grad_norm)
  - fine_model attribute (the current fine-tuned model)
"""
import numpy as np
import copy
from typing import Optional
from omegaconf import OmegaConf

import dgl
import torch
import flowmol
from torch.utils.data import ConcatDataset

from finetuning_solver.adjoint_matching.loss import (
    AMDataset,
    adj_matching_loss_list_of_dicts,
    create_timestep_subset,
)
from finetuning_solver.adjoint_matching.solver import LeanAdjointSolverFlow, step


MAX_ALLOWED_ATOMS = 75
MIN_ALLOWED_ATOMS = 30


def check_and_get_atom_numbers(config: OmegaConf):
    max_nodes = config.get("max_nodes", 210)
    if config.sampling.n_atoms is not None:
        if not isinstance(config.sampling.n_atoms, int):
            raise ValueError(f"n_atoms must be a positive int, got {config.sampling.n_atoms}")
        if config.sampling.n_atoms < MIN_ALLOWED_ATOMS or config.sampling.n_atoms > MAX_ALLOWED_ATOMS:
            raise ValueError(f"n_atoms must be between {MIN_ALLOWED_ATOMS} and {MAX_ALLOWED_ATOMS}, got {config.sampling.n_atoms}")
        if config.sampling.n_atoms * config.batch_size > max_nodes:
            raise ValueError(f"n_atoms * batch_size = {config.sampling.n_atoms * config.batch_size} > max_nodes ({max_nodes}). Please decrease n_atoms or increase max_nodes.")
        n_atoms = config.sampling.n_atoms
    else:
        n_atoms = None

    if config.sampling.min_num_atoms is not None:
        if not isinstance(config.sampling.min_num_atoms, int):
            raise ValueError(f"min_num_atoms must be a positive int, got {config.sampling.min_num_atoms}")
        if config.sampling.min_num_atoms < MIN_ALLOWED_ATOMS or config.sampling.min_num_atoms > MAX_ALLOWED_ATOMS:
            raise ValueError(f"min_num_atoms must be between {MIN_ALLOWED_ATOMS} and {MAX_ALLOWED_ATOMS}, got {config.sampling.min_num_atoms}")
        if config.sampling.min_num_atoms * config.batch_size > max_nodes:
            raise ValueError(f"min_num_atoms * batch_size = {config.sampling.min_num_atoms * config.batch_size} > max_nodes ({max_nodes}). Please decrease min_num_atoms or increase max_nodes.")
        min_num_atoms = config.sampling.min_num_atoms if n_atoms is None else None
    else:
        min_num_atoms = MIN_ALLOWED_ATOMS

    if config.sampling.max_num_atoms is not None:
        if not isinstance(config.sampling.max_num_atoms, int):
            raise ValueError(f"max_num_atoms must be a positive int, got {config.sampling.max_num_atoms}")
        if config.sampling.max_num_atoms < MIN_ALLOWED_ATOMS or config.sampling.max_num_atoms > MAX_ALLOWED_ATOMS:
            raise ValueError(f"max_num_atoms must be between {MIN_ALLOWED_ATOMS} and {MAX_ALLOWED_ATOMS}, got {config.sampling.max_num_atoms}")
        max_num_atoms = config.sampling.max_num_atoms if n_atoms is None else None
    else:
        max_num_atoms = MAX_ALLOWED_ATOMS

    return max_nodes, n_atoms, min_num_atoms, max_num_atoms


def sampling(
    config: OmegaConf,
    batch_size: int,
    model: flowmol.FlowMol,
    device: torch.device,
):
    model.to(device)
    n_atoms_provided = config.n_atoms is not None

    if n_atoms_provided:
        _, graph_trajectories = model.sample(
            sampler_type=config.sampler_type,
            n_atoms=torch.tensor([config.n_atoms] * batch_size, device=device),
            n_timesteps=config.num_integration_steps + 1,
            device=device,
            keep_intermediate_graphs=True,
        )
    else:
        _, graph_trajectories = model.sample_random_sizes(
            sampler_type=config.sampler_type,
            n_molecules=batch_size,
            n_timesteps=config.num_integration_steps + 1,
            device=device,
            min_num_atoms=config.min_num_atoms,
            max_num_atoms=config.max_num_atoms,
            keep_intermediate_graphs=True,
        )

    return graph_trajectories


class AdjointMatchingFinetuningTrainerFlowMol:
    def __init__(self,
            config: OmegaConf,
            model: flowmol.FlowMol,
            base_model: flowmol.FlowMol,
            grad_reward_fn: callable,
            device: torch.device = None,
            verbose: bool = False,
        ):
        # Config
        self.config = config
        self.sampling_config = config.sampling
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # Nodes limit
        (
            self.max_nodes,
            self.sampling_config.n_atoms,
            self.sampling_config.min_num_atoms,
            self.sampling_config.max_num_atoms,
        ) = check_and_get_atom_numbers(config)

        # Reward_lambda and LCT and clip_grad_norm
        reward_lambda = config.get("reward_lambda", 1.0)
        lct = config.get("lct", 0.0)
        self.LCT = lct * reward_lambda**2 if lct > 0.0 else 0.0
        self.clip_grad_norm = config.get("clip_grad_norm", 1.0)

        # Models
        self.fine_model = model
        self.base_model = base_model
        self.fine_model.to(self.device)
        self.base_model.to(self.device)

        # Reward (Gradient of the reward function(al))
        self.grad_reward_fn = grad_reward_fn
        self.features = list(config.get("features", ['x', 'a', 'c', 'e']))
        assert type(self.features) == list, f"features must be a list"

        # Engineering tricks:
        self.cutoff_time = config.get("cutoff_time", 0.5)
        samples_for_sumapproximation = config.get("samples_for_sumapproximation", 10)
        self.samples_for_sumapproximation = min(samples_for_sumapproximation, self.sampling_config.num_integration_steps + 1)
        self.final_percent = config.get("final_percent", 0.25)
        self.sample_percent = config.get("sample_percent", 0.25)

        # Setup optimizer
        self.configure_optimizers()

    def configure_optimizers(self):
        if hasattr(self, 'optimizer'):
            del self.optimizer
        self.optimizer = torch.optim.Adam(self.fine_model.parameters(), lr=self.config.lr)

    def get_model(self):
        return self.fine_model

    def sample_trajectories(self):
        self.fine_model.eval()

        graph_trajectories = sampling(
            config=self.sampling_config,
            batch_size=self.config.batch_size,
            model=self.fine_model,
            device=self.device,
        )

        ts = torch.linspace(0.0, 1.0, self.sampling_config.num_integration_steps + 1).to(self.device)
        sigmas = self.fine_model.vector_field.sigmas
        sigmas = torch.stack(sigmas, dim=0).to(self.device)
        return graph_trajectories, ts, sigmas

    def generate_dataset(self):
        """Sample dataset for training based on adjoint ODE and sampled trajectories."""
        datasets = []

        self.fine_model.eval()
        self.base_model.eval()

        solver = LeanAdjointSolverFlow(self.base_model, self.grad_reward_fn)

        iterations = self.sampling_config.num_samples // self.config.batch_size
        avg_adj_0_norm = 0.0
        for i in range(iterations):
            with torch.no_grad():
                while True:
                    graph_trajectories, ts, sigmas = self.sample_trajectories()
                    if self.verbose:
                        print(f"Sampled trajectory with {graph_trajectories[0].num_nodes()} nodes and {graph_trajectories[0].num_edges()} edges.")
                    if graph_trajectories[0].num_nodes() <= self.max_nodes:
                        break
                    if self.verbose:
                        print(f"Rerolling: got {graph_trajectories[0].num_nodes()} nodes (> {self.max_nodes})")

            ts = ts.flip(0)
            sigmas = sigmas.flip(0)
            graph_trajectories = graph_trajectories[::-1]

            if self.cutoff_time > 0.0:
                cutoff_idx = int((1 - self.cutoff_time) * (ts.shape[0] - 1))
                ts = ts[:cutoff_idx + 1]
                graph_trajectories = graph_trajectories[:cutoff_idx + 1]
                sigmas = sigmas[:cutoff_idx]

            solver_info, adj_0_norm = solver.solve(graph_trajectories=graph_trajectories, ts=ts)
            solver_info['sigma_t'] = sigmas

            dataset = AMDataset(solver_info=solver_info)
            datasets.append(dataset)
            avg_adj_0_norm += adj_0_norm

        if len(datasets) == 0:
            return None, None
        avg_adj_0_norm /= len(datasets)
        dataset = ConcatDataset(datasets)
        return dataset, avg_adj_0_norm

    def push_to_device(self, sample):
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value.to(self.device)
            elif isinstance(value, list):
                if isinstance(value[0], dgl.DGLGraph):
                    for i in range(len(value)):
                        value[i] = value[i].to(self.device)
                if isinstance(value[0], dict):
                    for dict_i in range(len(value)):
                        for k, v in value[dict_i].items():
                            if isinstance(v, torch.Tensor):
                                value[dict_i][k] = v.to(self.device)
        return sample

    def train_step(self, sample):
        """Training step."""
        sample = self.push_to_device(sample)
        ts = sample['t']
        sigmas = sample['sigma_t']
        alpha = sample['alpha']
        alpha_dot = sample['alpha_dot']
        traj_g = sample['traj_graph']
        traj_adj = sample['traj_adj']
        traj_v_base = sample['traj_v_base']

        idxs = create_timestep_subset(ts.shape[0], self.final_percent, self.sample_percent, self.samples_for_sumapproximation)

        v_base = []
        v_fine = []
        adj = []
        sigma = []

        dt = ts[0] - ts[1]

        for idx in idxs:
            t = ts[idx]
            sigma_t = sigmas[idx]
            adj_t = traj_adj[idx]
            v_base_t = traj_v_base[idx]
            g_base_t = traj_g[idx]
            alpha_t = alpha[idx]
            alpha_dot_t = alpha_dot[idx]

            v_fine_t, _ = step(
                model=self.fine_model,
                adj=None,
                g_t=g_base_t,
                t=t,
                alpha=alpha_t,
                alpha_dot=alpha_dot_t,
                dt=dt,
                upper_edge_mask=g_base_t.edata['ue_mask'],
                calc_adj=False,
            )

            v_base.append(v_base_t)
            v_fine.append(v_fine_t)
            adj.append(adj_t)
            sigma.append(sigma_t)

        assert len(v_base) == len(v_fine) == len(adj) == len(sigma)

        v_base = {feat: torch.stack([v_base[i][feat] for i in range(len(v_base))], dim=0) for feat in self.features}
        v_fine = {feat: torch.stack([v_fine[i][feat] for i in range(len(v_fine))], dim=0) for feat in self.features}
        adj = {feat: torch.stack([adj[i][feat] for i in range(len(adj))], dim=0) for feat in self.features}
        sigma = torch.stack(sigma, dim=0)

        loss = adj_matching_loss_list_of_dicts(
            v_base=v_base,
            v_fine=v_fine,
            adj=adj,
            sigma=sigma,
            LCT=self.LCT,
            features=self.features
        )

        if loss.isnan().any():
            return torch.tensor(float("inf"), device=self.device)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=False)

        grads = []
        for p in self.fine_model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        grad_vec = torch.cat(grads)
        grad_norm = grad_vec.norm(2)
        if self.verbose:
            print(f"L2 norm of Gradient: {grad_norm.item():.6f}")

        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.fine_model.parameters(), self.clip_grad_norm)

        self.optimizer.step()

        return loss.item(), grad_norm.item()

    def finetune(self, dataset, steps=None):
        """Finetuning the model."""
        c = 0
        total_loss = 0
        total_grad_norm = 0

        self.fine_model.to(self.device)
        self.fine_model.train()

        self.optimizer.zero_grad()

        if steps is not None:
            idxs = np.random.permutation(dataset.__len__())[:steps]
        else:
            idxs = np.random.permutation(dataset.__len__())

        for idx in idxs:
            sample = dataset[idx]
            loss, grad_norm = self.train_step(sample)
            total_loss = total_loss + loss
            total_grad_norm = total_grad_norm + grad_norm
            c += 1

        del dataset
        return total_loss / c, total_grad_norm / c
