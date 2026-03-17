"""
Adjoint Matching loss functions and utilities.

These are model-agnostic components of the Adjoint Matching algorithm
(Domingo-Enrich et al., 2025). They can be reused with any flow/diffusion model.
"""
import numpy as np
import torch
import dgl
from torch.utils.data import Dataset


def create_timestep_subset(
        total_steps,
        final_percent: float = 0.25,
        sample_percent: float = 0.25,
        samples_for_sumapproximation: int = None,
    ) -> np.ndarray:
    """
    Create a subset of time-steps for efficient computation. (See AM-Paper Appendix G2)

    Args:
        total_steps (int): Total number of time-steps in the process
        final_percent (float): Percentage of final steps to always include
        sample_percent (float): Percentage of additional steps to sample
        samples_for_sumapproximation (int, optional): Maximum number of steps to include
            in the final subset. If None, includes all selected steps.

    Returns:
        np.ndarray: Sorted array of selected timestep indices
    """
    final_steps_count = int(total_steps * final_percent)
    sample_steps_count = int(total_steps * sample_percent)

    # Always take the first final_percent steps
    final_samples = np.arange(final_steps_count)

    # Sample additional steps without replacement from the remaining steps
    remaining_steps = np.setdiff1d(np.arange(total_steps), final_samples)
    additional_samples = np.random.choice(
        remaining_steps,
        size=sample_steps_count,
        replace=False
    )
    combined_samples = np.sort(np.concatenate([final_samples, additional_samples]))

    # Take at most samples_for_sumapproximation samples
    if samples_for_sumapproximation is not None and samples_for_sumapproximation < combined_samples.shape[0]:
        combined_samples = np.random.choice(
            combined_samples,
            size=samples_for_sumapproximation,
            replace=False
        )

    return np.sort(combined_samples)


class AMDataset(Dataset):
    def __init__(self, solver_info):
        solver_info = self.detach_all(solver_info)
        self.t = solver_info['t']
        self.sigma_t = solver_info['sigma_t']
        self.alpha = solver_info['alpha']
        self.alpha_dot = solver_info['alpha_dot']
        self.traj_g = solver_info['traj_graph']
        self.traj_adj = solver_info['traj_adj']
        self.traj_v_base = solver_info['traj_v_base']

        self.T = self.t.size(0)
        self.bs = 1

    def __len__(self):
        return self.bs

    def __getitem__(self, idx):
        return {
            't': self.t,
            'sigma_t': self.sigma_t,
            'alpha': self.alpha,
            'alpha_dot': self.alpha_dot,
            'traj_graph': self.traj_g,
            'traj_adj': self.traj_adj,
            'traj_v_base': self.traj_v_base,
        }

    def detach_all(self, solver_info):
        for key, value in solver_info.items():
            if isinstance(value, torch.Tensor):
                solver_info[key] = value.detach()
            elif isinstance(value, list):
                if isinstance(value[0], dgl.DGLGraph):
                    for g in value:
                        for k in g.ndata.keys():
                            if isinstance(g.ndata[k], torch.Tensor):
                                g.ndata[k] = g.ndata[k].detach()
                        for k in g.edata.keys():
                            if isinstance(g.edata[k], torch.Tensor):
                                g.edata[k] = g.edata[k].detach()
                if isinstance(value[0], dict):
                    for dict_i in range(len(value)):
                        for k, v in value[dict_i].items():
                            if isinstance(v, torch.Tensor):
                                value[dict_i][k] = v.detach()
        return solver_info


def adj_matching_loss(v_base, v_fine, adj, sigma, LCT):
    """Adjoint matching loss for FM"""
    eps = 1e-12
    diff = v_fine - v_base
    sig = sigma.view(-1, 1, 1)
    term_diff = (2.0 / (sig + eps)) * diff
    term_adj = sig * adj
    term = term_diff + term_adj
    per_t = (term ** 2).sum(dim=[1, 2])
    clipped = torch.clamp(per_t, max=LCT) if LCT > 0.0 else per_t
    loss = clipped.sum()
    return loss


loss_weights = {'a': 0.4, 'c': 1.0, 'e': 2.0, 'x': 3.0}


def adj_matching_loss_list_of_dicts(v_base, v_fine, adj, sigma, LCT, features=['x', 'a', 'c', 'e']):
    """Adjoint matching loss for FM"""
    eps = 1e-12
    loss = 0.0
    for i, feat in enumerate(features):
        diff = v_fine[feat] - v_base[feat]
        sig = sigma[:, i].view(-1, 1, 1)
        term_diff = (2.0 / (sig + eps)) * diff
        term_adj = sig * adj[feat]
        term = term_diff + term_adj
        per_t = (term ** 2).sum(dim=[1, 2])
        clipped = torch.clamp(per_t, max=LCT) if LCT > 0.0 else per_t
        loss = loss + clipped.sum() * loss_weights[feat]
    return loss
