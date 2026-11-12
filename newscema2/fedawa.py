from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.common import (
    FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _flatten(ndarrays):
    return np.concatenate([a.ravel() for a in ndarrays]).astype(np.float32)


def _weighted_average(models, weights):
    out = []
    for i in range(len(models[0])):
        stack = np.stack([m[i] for m in models], axis=0).astype(np.float64)
        w = weights.reshape([-1] + [1] * (stack.ndim - 1))
        out.append(np.sum(stack * w, axis=0).astype(models[0][i].dtype))
    return out


def _cost_vector(g, locals_, dis):
    """g: (D,), locals_: (N, D) -> (N,)"""
    if dis == "cos":
        return 1 - torch.nn.functional.cosine_similarity(
            g.unsqueeze(0), locals_, dim=-1, eps=1e-8)
    return torch.mean(torch.abs(g.unsqueeze(0) - locals_) ** 2, dim=-1)  # 'euc'


def build_fedawa_strategy(
    num_clients, init_params, evaluate_fn, fit_metrics_agg,
    server_epochs=1, server_optimizer="adam", reg_distance="cos",
    gamma=1.0, device=None,
):
    return FedAWA(
        server_epochs=server_epochs, server_optimizer=server_optimizer,
        reg_distance=reg_distance, gamma=gamma, device=device,
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=num_clients, min_available_clients=num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )


class FedAWA(FedAvg):
    def __init__(self, *args, server_epochs=1, server_optimizer="adam",
                 reg_distance="cos", gamma=1.0, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_epochs = server_epochs
        self.server_optimizer = server_optimizer
        self.reg_distance = reg_distance
        self.gamma = gamma
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        init = kwargs.get("initial_parameters")
        self._global_ndarrays: Optional[List[np.ndarray]] = (
            parameters_to_ndarrays(init) if init is not None else None
        )
        # logit bobot per client (persisten antar round), key = cid
        self._T: Dict[str, float] = {}
        self.last_weights: Dict[str, float] = {}

    def _learn_weights(self, cids, deltas_t, g_flat_t, local_t, size_w):
        # inisialisasi client baru dengan bobot ukuran data (seperti kode asli)
        for c, w in zip(cids, size_w):
            self._T.setdefault(c, float(w))

        T = torch.tensor([self._T[c] for c in cids], dtype=torch.float32,
                         device=self.device, requires_grad=True)

        if self.server_optimizer == "sgd":
            opt = torch.optim.SGD([T], lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            opt = torch.optim.Adam([T], lr=0.001, betas=(0.5, 0.999))

        C = _cost_vector(g_flat_t, local_t, self.reg_distance).detach()

        for _ in range(self.server_epochs):
            p = torch.softmax(T, dim=0)
            reg_loss = torch.sum(p * C)
            col = p.unsqueeze(0) @ deltas_t                  # (1, D)
            l2 = torch.norm(deltas_t - col, p=2, dim=1)      # (N,)
            sim_loss = torch.sum(p * l2)
            loss = sim_loss + reg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        for c, v in zip(cids, T.detach().cpu().tolist()):
            self._T[c] = v
        return torch.softmax(T.detach(), dim=0).cpu().numpy().astype(np.float64)

    def aggregate_fit(self, server_round, results: List[Tuple[ClientProxy, FitRes]], failures):
        if not results:
            return None, {}

        cids = [proxy.cid for proxy, _ in results]
        client_nd = [parameters_to_ndarrays(r.parameters) for _, r in results]
        n_ex = np.array([r.num_examples for _, r in results], dtype=np.float64)
        size_w = n_ex / n_ex.sum()

        if self._global_ndarrays is None:           # fallback round 1
            self._global_ndarrays = client_nd[0]

        g_flat = _flatten(self._global_ndarrays)
        local = np.stack([_flatten(c) for c in client_nd])
        local_t = torch.from_numpy(local).to(self.device)
        g_t = torch.from_numpy(g_flat).to(self.device)
        deltas_t = local_t - g_t

        p = self._learn_weights(cids, deltas_t, g_t, local_t, size_w)

        # global_baru = global + gamma * sum_i p_i * (theta_i - global)
        avg = _weighted_average(client_nd, p)
        aggregated = [
            (g + self.gamma * (a.astype(np.float64) - g)).astype(g.dtype)
            for g, a in zip(self._global_ndarrays, avg)
        ]

        self._global_ndarrays = aggregated
        params = ndarrays_to_parameters(aggregated)

        metrics: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn is not None:
            metrics = self.fit_metrics_aggregation_fn(
                [(r.num_examples, r.metrics) for _, r in results])
        self.last_weights = {c: float(w) for c, w in zip(cids, p)}
        for (_, r), c, w in zip(results, cids, p):
            name = r.metrics.get("client_name", c)
            metrics[f"awa_weight_{name}"] = float(w)
        return params, metrics