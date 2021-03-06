# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Implementing t-SNE

# ## Dataset

# +
from scipy.spatial import distance
from sklearn import datasets

digits = datasets.load_digits()
dist_matrix = distance.cdist(digits.data, digits.data)
# -

# ## scikit-learn's Reference Implementation

# +
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, metric='precomputed', method='exact', random_state=42).fit_transform(dist_matrix)

import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=digits.target)
plt.colorbar()
# -

# ## My Own Implementation

# +
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyTSNE(nn.Module):
    def __init__(self,
            d: torch.Tensor,
            n_components: int,
            perplexity: float,
            random_state: Optional[int]=None):
        super().__init__()

        assert d.shape[0] == d.shape[1]
        self.n = d.shape[0]

        # Perplexity must be less than N - 1 because \lim_{\sigma \to +\infty} Perp(P_i) = N - 1
        if perplexity >= self.n - 1:
            raise ValueError('Perplexity must be less than N - 1')
        self.perplexity = perplexity

        if random_state is not None:
            torch.manual_seed(random_state)
        self.embeddings = nn.Parameter(torch.normal(0, 1e-4, (self.n, n_components)))

        # sigma
        sigma = MyTSNE.find_sigma(d, perplexity)

        # p_cond
        r = torch.exp(-d ** 2 / (2 * sigma.reshape(-1, 1) ** 2))
        p_cond = r / (r.sum(dim=1) - 1).reshape(-1, 1)  # p_cond[i, j] is p_{j|i}

        # p
        p = (p_cond + p_cond.transpose(0, 1)) / (2 * self.n)
        self.register_buffer('p', p)

    def forward(self):
        device = next(self.parameters()).device
        # q
        ys = self.embeddings
        d_low = torch.cdist(ys, ys)
        r = (1 + d_low ** 2) ** -1
        del d_low
        q = r / (r.sum() - self.n)  # q[i, j] is q_{ij}
        del r
        # KL divergence
        p = self.p
        c = p * (torch.log(p) - torch.log(q))
        c.fill_diagonal_(0)
        C = c.sum()
        return C

    def find_sigma(d, target_perp):
        """Find an optimal sigma for each i by using binary search"""
        assert d.shape[0] == d.shape[1]
        n = d.shape[0]

        from tqdm import tqdm
        sigma = torch.zeros(n)
        for i in tqdm(range(n)):
            def f(x):
                return MyTSNE.compute_perp_i(i, d[i, :], x) - target_perp
            lb, ub = MyTSNE.find_initial_bounds(f)
            sigma[i] = MyTSNE.binary_search(f, lb, ub)
        return sigma

    def compute_perp_i(i, d_i, sigma_i):
        """Compute the perplexity for the i-th point with a given sigma"""
        # p_i
        r_i = torch.exp(-d_i ** 2 / (2 * sigma_i ** 2))
        p_i = r_i / (r_i.sum() - 1)
        p_i[i] = 1
        # H(p_i)
        h_p_i = -(p_i * torch.log2(p_i)).sum()
        # Perp_i
        perp_i = 2 ** h_p_i
        return perp_i

    def find_initial_bounds(f):
        """Find lower and upper bounds for the solution of f(x) = 0

        Assumptions:
        - x > 0
        - `f` is monotonically increasing
        """
        x = 1
        y = f(x)
        if y > 0:
            # Current x is too large
            ub = x
            # Find a lower bound that makes y < 0
            while y >= 0:
                x /= 2
                y = f(x)
            lb = x
        elif y < 0:
            # Current x is too small
            lb = x
            # Find a upper bound that makes y > 0
            while y <= 0:
                x *= 2
                y = f(x)
            ub = x
        else:
            # y == 0
            ub = x
            lb = x
        return lb, ub

    def binary_search(f, lb, ub):
        """Find the solution of f(x) = 0 with in [lb, ub]"""
        while True:
            x = (lb + ub) / 2

            if x == lb or x == ub:
                # Reached the precision limit
                return x

            y = f(x)
            if y > 0:
                # Current x is too large
                ub = x
            elif y < 0:
                # Current x is too small
                lb = x
            else:
                # Found the solution
                return x

def embed(
        d: np.ndarray,
        n_components: int=2,
        *,
        perplexity: float=30.,
        optimizer_class: Type[optim.Optimizer]=optim.Adam,
        optimizer_kwargs: Dict[str, Any]={ 'lr': 1. },
        n_iter: int=1000,
        random_state: Optional[int]=None,
        device: torch.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) -> np.ndarray:
    d = d / d.max()
    d = torch.from_numpy(d)
    net = MyTSNE(d, n_components, perplexity, random_state).to(device=device, dtype=torch.float32)
    optimizer = optimizer_class(net.parameters(), **optimizer_kwargs)
    from tqdm import tqdm
    for epoch in tqdm(range(n_iter)):
        optimizer.zero_grad()
        loss = net()
        loss.backward()
        optimizer.step()
    return net.embeddings.cpu().detach().numpy()

X_embedded = embed(dist_matrix, n_components=2, random_state=42)

import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=digits.target)
plt.colorbar()

import gc
gc.collect()
torch.cuda.empty_cache()
