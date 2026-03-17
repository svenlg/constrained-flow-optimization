import random
import numpy as np
import torch
import dgl


def set_seed(seed: int):
    """Seed all random generators."""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
