from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _flatten(ndarrays: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([arr.ravel() for arr in ndarrays])


def _weighted_average(
    models: List[List[np.ndarray]], weights: np.ndarray
) -> List[np.ndarray]:
    """Weighted average per layer. weights harus sudah sum = 1."""
    out = []
    for layer_idx in range(len(models[0])):
        stack = np.stack([m[layer_idx] for m in models], axis=0).astype(np.float64)
        w = weights.reshape([-1] + [1] * (stack.ndim - 1))
        out.append(np.sum(stack * w, axis=0).astype(models[0][layer_idx].dtype))
    return out


# ============================================================
# Builder (signature sama dengan build_fedcosine_pairwise_strategy)
# ============================================================
def build_fedsim_strategy(
    num_clients: int,
    init_params: Parameters,
    evaluate_fn,
    fit_metrics_agg,
    num_groups: int = 3,
    pca_variance: float = 0.95,
    group_weighting: str = "uniform",
    seed: int = 0,
) -> "FedSim":
    return FedSim(
        num_groups=num_groups,
        pca_variance=pca_variance,
        group_weighting=group_weighting,
        seed=seed,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )


# ============================================================
# FedSim Strategy
# ============================================================
class FedSim(FedAvg):
    """
    FedSim di atas Flower.

    Alur aggregate_fit:
        1. Hitung update tiap client: delta = theta_client - theta_global.
        2. Round 1: client dibagi round-robin ke grup.
           Round > 1: delta -> PCA (pca_variance) -> KMeans(num_groups).
        3. Intra-group: FedAvg berbobot num_examples di tiap grup.
        4. Inter-group: rata-rata antar model grup
             "uniform" -> sederhana (persis FedSim asli)
             "samples" -> berbobot total sampel grup (opsional)
        5. Hasil jadi model global baru.

    Catatan: FedSim asli mengcluster memakai gradient (get_grads) sebelum
    training. Di Flower server hanya menerima hasil fit, jadi delta
    parameter dipakai sebagai pengganti gradient.
    """

    def __init__(
        self,
        *args,
        num_groups: int = 3,
        pca_variance: float = 0.95,
        group_weighting: str = "uniform",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if num_groups < 1:
            raise ValueError("num_groups harus >= 1")
        if group_weighting not in ("uniform", "samples"):
            raise ValueError("group_weighting harus 'uniform' atau 'samples'")

        self.num_groups = num_groups
        self.pca_variance = pca_variance
        self.group_weighting = group_weighting
        self.seed = seed

        # Salinan model global (FedAvg menghapus initial_parameters setelah dipakai)
        init = kwargs.get("initial_parameters")
        self._global_ndarrays: Optional[List[np.ndarray]] = (
            parameters_to_ndarrays(init) if init is not None else None
        )

        # Untuk logging/analisis dari luar
        self.last_labels: Optional[np.ndarray] = None
        self.last_group_sizes: Dict[int, int] = {}

    # --------------------------------------------------------
    def _cluster(self, deltas: np.ndarray, server_round: int) -> np.ndarray:
        n = deltas.shape[0]
        k = min(self.num_groups, n)

        # Round pertama: round-robin
        if server_round == 1 or self._global_ndarrays is None:
            return np.arange(n) % k

        if k == 1 or n < 2:
            return np.zeros(n, dtype=int)

        pca = PCA(n_components=self.pca_variance, svd_solver="full")
        reduced = pca.fit_transform(deltas)

        km = KMeans(n_clusters=k, n_init=10, random_state=self.seed + server_round)
        km.fit(reduced)
        return km.labels_

    # --------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        if not results:
            return None, {}

        client_ndarrays = [parameters_to_ndarrays(r.parameters) for _, r in results]
        num_examples = np.array([r.num_examples for _, r in results], dtype=np.float64)

        # Pada round pertama, global belum diketahui kalau tidak diberikan
        if self._global_ndarrays is not None:
            g_flat = _flatten(self._global_ndarrays)
            deltas = np.stack([_flatten(c) - g_flat for c in client_ndarrays])
        else:
            deltas = np.stack([_flatten(c) for c in client_ndarrays])

        # ---- Clustering ----
        labels = self._cluster(deltas, server_round)
        self.last_labels = labels
        group_ids = sorted(set(labels.tolist()))

        # ---- Intra-group: FedAvg berbobot sampel ----
        group_models: List[List[np.ndarray]] = []
        group_samples: List[float] = []
        self.last_group_sizes = {}
        for g in group_ids:
            idx = np.where(labels == g)[0]
            w = num_examples[idx] / num_examples[idx].sum()
            group_models.append(_weighted_average([client_ndarrays[i] for i in idx], w))
            group_samples.append(num_examples[idx].sum())
            self.last_group_sizes[int(g)] = len(idx)

        # ---- Inter-group ----
        # Dibagi jumlah grup yang BENAR-BENAR ada (bukan num_groups tetap)
        if self.group_weighting == "uniform":
            gw = np.full(len(group_models), 1.0 / len(group_models))
        else:
            gs = np.array(group_samples)
            gw = gs / gs.sum()
        aggregated = _weighted_average(group_models, gw)

        self._global_ndarrays = aggregated
        aggregated_parameters = ndarrays_to_parameters(aggregated)

        # ---- Metrics ----
        aggregated_metrics: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn is not None:
            aggregated_metrics = self.fit_metrics_aggregation_fn(
                [(r.num_examples, r.metrics) for _, r in results]
            )

        aggregated_metrics["num_active_groups"] = len(group_ids)
        for pos, g in enumerate(group_ids):
            aggregated_metrics[f"group_{g}_size"] = int(self.last_group_sizes[int(g)])
            aggregated_metrics[f"group_{g}_weight"] = float(gw[pos])
        for idx, (_, r) in enumerate(results):
            name = r.metrics.get("client_name", f"client_{idx}")
            aggregated_metrics[f"cluster_{name}"] = int(labels[idx])

        return aggregated_parameters, aggregated_metrics