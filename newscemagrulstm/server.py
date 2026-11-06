"""Sisi SERVER Flower: daftar algoritma, mixin bandwidth/log, make_strategy, server_fn."""
import inspect
import random

import numpy as np
import pandas as pd
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

import config
from data_utils import get_in_dim, load_all
from evaluation import weighted_metrics
from model import build_model
from utils import _cfg, _finite, get_seed, get_weights, nbytes

# FedCosinePairwise opsional: kalau file tidak ada, algoritma lain tetap jalan
try:
    from FedCosinePairwise import FedCosinePairwise
except Exception as _e:                      # ImportError / error lain di file itu
    FedCosinePairwise = None
    print(f"[peringatan] FedCosinePairwise tidak bisa diimpor ({_e!r}); "
          f"varian 'fedpairwise' dinonaktifkan")

# Fedsimstrategy opsional: kalau file tidak ada, algoritma lain tetap jalan
try:
    from Fedsimstrategy import FedSim
except Exception as _e:
    FedSim = None
    print(f"[peringatan] Fedsimstrategy tidak bisa diimpor ({_e!r}); "
          f"varian 'fedsim' dinonaktifkan")

# Varian fedpairwise: nama -> (temperature, data_influence)
PW_VARIANTS = {
    'fedpairwise_T0.01_d0.3': (0.01, 0.3),
    'fedpairwise_T0.01_d0.5': (0.01, 0.5),
    'fedpairwise_T0.01_d0':   (0.01, 0.0),
}

# Varian fedsim: nama -> (num_groups, group_weighting)
FS_VARIANTS = {
    'fedsim_g3': (3, 'uniform'),
    'fedsim_g5': (5, 'uniform'),
}

ALGOS = ('fedavg', 'fedprox', *PW_VARIANTS, *FS_VARIANTS)


# ============================================================
# STRATEGY: mixin pencatat bandwidth + metrik val per ronde
# ============================================================
class BandwidthMixin:
    """
    Dipasang di depan strategy Flower apa pun.
    Log tiap ronde -> OUT_DIR/ronde_log.csv   (val_loss + 6 metrik val berbobot)
    Log per client -> OUT_DIR/ronde_val_per_client.csv
    Bobot global terakhir -> OUT_DIR/final_params.npz
    (Log per ronde = client TRAINING saja; evaluasi semua client ada di full_report.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_fit = self.up_fit = self.down_eval = 0
        self.log = []
        self.log_client = []
        self.latest_weights = None

    def configure_fit(self, server_round, parameters, client_manager):
        cfg = super().configure_fit(server_round, parameters, client_manager)
        self.down_fit = len(cfg) * nbytes(parameters_to_ndarrays(parameters))
        return cfg

    def aggregate_fit(self, server_round, results, failures):
        config.ensure_out_dir()
        if failures:
            print(f"[ronde {server_round:02d}] {len(failures)} client gagal "
                  f"(lihat {config.OUT_DIR / 'client_error.log'})")
        ok = []
        for cp, r in results:
            if _finite(parameters_to_ndarrays(r.parameters)):
                ok.append((cp, r))
            else:
                print(f"[ronde {server_round:02d}] bobot NaN dari satu client dibuang")
        if not ok:
            return None, {}
        results = ok

        self.up_fit = sum(nbytes(parameters_to_ndarrays(r.parameters))
                          for _, r in results)
        agg, metrics = super().aggregate_fit(server_round, results, failures)
        if agg is not None:
            self.latest_weights = parameters_to_ndarrays(agg)
            np.savez(config.OUT_DIR / 'final_params.npz', *self.latest_weights)

        sim = getattr(self, 'last_similarity_matrix', None)
        if sim is not None:
            np.savetxt(config.OUT_DIR / f'sim_matrix_r{server_round:02d}.csv',
                       sim, delimiter=',')
            names = [r.metrics.get('client_name', f'client_{i}')
                     for i, (_, r) in enumerate(results)]
            w = [metrics.get(f'agg_weight_{n}', float('nan')) for n in names]
            print(f"[ronde {server_round:02d}] bobot agregasi: "
                  + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, w)))
        return agg, metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        cfg = super().configure_evaluate(server_round, parameters, client_manager)
        self.down_eval = len(cfg) * nbytes(parameters_to_ndarrays(parameters))
        return cfg

    def aggregate_evaluate(self, server_round, results, failures):
        config.ensure_out_dir()
        if not results:
            return None, {}
        loss, _ = super().aggregate_evaluate(server_round, results, failures)
        m = weighted_metrics([(r.num_examples, r.metrics) for _, r in results])

        # simpan metrik val tiap client per ronde
        for _, r in results:
            self.log_client.append({'round': server_round,
                                    'n_val': r.num_examples,
                                    'val_loss': r.loss, **r.metrics})
        pd.DataFrame(self.log_client).to_csv(
            config.OUT_DIR / 'ronde_val_per_client.csv', index=False)

        down = self.down_fit + self.down_eval
        up = self.up_fit
        self.log.append({'round': server_round, 'val_loss': loss, **m,
                         'n_client_fit': len(results),
                         'down_MB': down / 1e6, 'up_MB': up / 1e6,
                         'total_MB': (down + up) / 1e6})
        pd.DataFrame(self.log).to_csv(config.OUT_DIR / 'ronde_log.csv', index=False)
        print(f"[ronde {server_round:02d}] VAL "
              f"acc {m['acc']:.4f} | prec {m['prec']:.4f} | rec {m['rec']:.4f} | "
              f"F1 {m['f1']:.4f} | AUC {m['auc']:.4f} | PR-AUC {m['prauc']:.4f} "
              f"| bw {(down + up) / 1e6:.3f} MB")
        return loss, m


def _accepted_kwargs(cls, kwargs):
    """Saring kwargs sesuai signature cls.__init__.
    Kalau cls menerima **kwargs, semua dikirim. Mengembalikan (diterima, dibuang)."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs), []
    ok = {k: v for k, v in kwargs.items() if k in params}
    dropped = [k for k in kwargs if k not in params]
    return ok, dropped


def make_strategy(algo, init, n_clients, model):
    """Bangun strategy sesuai nama algoritma."""
    if algo not in ALGOS:
        # toleransi huruf besar/kecil (nama varian memuat 'T')
        alias = {a.lower(): a for a in ALGOS}
        algo = alias.get(str(algo).lower(), algo)
    if algo not in ALGOS:
        raise ValueError(f"algo '{algo}' tidak dikenal, pilih salah satu: {ALGOS}")

    mu = float(_cfg('FEDPROX_MU', 0.01)) if algo == 'fedprox' else 0.0

    def fit_cfg(rnd):
        return {'algo': algo, 'mu': mu, 'round': rnd, 'seed': get_seed()}

    common = dict(
        fraction_fit=config.FRACTION_FIT,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=n_clients,
        initial_parameters=init,
        on_fit_config_fn=fit_cfg,
    )

    # ---------- FedAvg / FedProx ----------
    if algo in ('fedavg', 'fedprox'):
        cls = type('FedAvgBandwidth', (BandwidthMixin, FedAvg), {})
        return cls(**common)

    # ---------- FedSim ----------
    if algo in FS_VARIANTS:
        if FedSim is None:
            raise ImportError("Fedsimstrategy.FedSim tidak tersedia; "
                              "pastikan Fedsimstrategy.py ada di folder yang sama.")
        n_groups, g_weighting = FS_VARIANTS[algo]
        if n_groups > n_clients:
            raise ValueError(f"{algo}: num_groups={n_groups} > jumlah client "
                             f"training ({n_clients}). Tambah state di FL_STATES "
                             f"atau pakai varian dengan grup lebih sedikit.")
        cls = type('FedSimBandwidth', (BandwidthMixin, FedSim), {})

        fs = dict(num_groups=int(n_groups), group_weighting=g_weighting)
        # argumen opsional dari config (hanya dikirim kalau FedSim menerimanya)
        for name, key in (('seed', None), ('use_delta', 'FEDSIM_USE_DELTA')):
            if name == 'seed':
                fs['seed'] = get_seed()
            elif _cfg(key, None) is not None:
                fs[name] = _cfg(key, None)

        fs_ok, dropped = _accepted_kwargs(FedSim, fs)
        if dropped:
            print(f"[peringatan] FedSim tidak menerima argumen {dropped}; diabaikan. "
                  f"Sesuaikan nama argumen di make_strategy bila perlu.")
        return cls(**fs_ok, **common)

    # ---------- FedPairwise ----------
    if algo in PW_VARIANTS:
        if FedCosinePairwise is None:
            raise ImportError("FedCosinePairwise tidak tersedia; "
                              "pastikan FedCosinePairwise.py ada di folder yang sama.")
        temp, d_inf = PW_VARIANTS[algo]
        cls = type('FedPairwiseBandwidth', (BandwidthMixin, FedCosinePairwise), {})
        pw = dict(temperature=float(temp),
                  min_weight=float(_cfg('PAIRWISE_MIN_WEIGHT', 0.0)),
                  similarity_agg=_cfg('PAIRWISE_SIM_AGG', 'mean'),
                  data_influence=float(d_inf))
        if 'use_delta' in inspect.signature(FedCosinePairwise.__init__).parameters:
            pw['use_delta'] = bool(_cfg('PAIRWISE_USE_DELTA', True))
        else:
            print("[peringatan] FedCosinePairwise versi lama: use_delta diabaikan "
                  "(similarity dihitung pada bobot penuh)")
        return cls(**pw, **common)

    raise ValueError(f"algo '{algo}' tidak dikenal, pilih salah satu: {ALGOS}")


# ============================================================
# SERVER
# ============================================================
def server_fn(context, algo=None) -> ServerAppComponents:
    config.ensure_out_dir()
    algo = algo or _cfg('ALGO', 'fedavg')
    seed = get_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_clients = len(load_all())
    model = build_model(get_in_dim(), seed=get_seed())
    init = ndarrays_to_parameters(get_weights(model))

    print(f"[server] algoritma agregasi: {algo} | seed: {seed}")
    strategy = make_strategy(algo, init, n_clients, model)
    return ServerAppComponents(strategy=strategy,
                               config=ServerConfig(num_rounds=config.NUM_ROUNDS))


def make_server_app(algo=None) -> ServerApp:
    return ServerApp(server_fn=lambda ctx: server_fn(ctx, algo))