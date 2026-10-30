import inspect
import os
import random
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import (Context, NDArrays, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import (FedAdagrad, FedAdam, FedAvg, FedAvgM,
                                  FedYogi)
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import config
from dataset import prepare_client_data
from model import build_model

# aggregation4 opsional: kalau file tidak ada, algoritma lain tetap jalan
try:
    from aggregation4 import FedCosinePairwise
except Exception as _e:                      # ImportError / error lain di file itu
    FedCosinePairwise = None
    print(f"[peringatan] aggregation4 tidak bisa diimpor ({_e!r}); "
          f"varian 'fedpairwise' dinonaktifkan")

# fedsim_strategy opsional: kalau file tidak ada, algoritma lain tetap jalan
try:
    from Fedsimstrategy import FedSim
except Exception as _e:
    FedSim = None
    print(f"[peringatan] fedsim_strategy tidak bisa diimpor ({_e!r}); "
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

# Metrik lengkap (tidak bergantung pada config.METRIK yang mungkin hanya f1/auc)
METRIK = ['acc', 'prec', 'rec', 'f1', 'auc', 'prauc']
NAMA_METRIK = {
    'acc': 'Accuracy', 'prec': 'Precision', 'rec': 'Recall',
    'f1': 'F1-score', 'auc': 'ROC-AUC', 'prauc': 'PR-AUC',
}


def _cfg(name, default):
    return getattr(config, name, default)


def get_seed():
    """Seed aktif: FL_SEED (diset main.py) menang atas config.SEED."""
    v = os.environ.get('FL_SEED')
    return int(v) if v not in (None, '') else int(config.SEED)


# ============================================================
# 1. UTILITAS & DATA
#    - load_data()      : data TRAINING  (hanya state di FL_STATES)
#    - load_eval_data() : data EVALUASI  (SEMUA client, val & test)
# ============================================================
def _wanted_states():
    raw = os.environ.get('FL_STATES', '').strip()
    return {int(x) for x in raw.split(',') if x} or None


@lru_cache(maxsize=1)
def _load_raw():
    """Semua client apa adanya (scaler dari semua state). Di-cache sekali per proses."""
    d = os.environ.get('FL_CLIENT_DIR') or config.CLIENT_DIR
    return prepare_client_data(Path(d))


def _rescale_client(c, old, new):
    """Kembalikan X ke skala asli (scaler lama), lalu scale dengan scaler baru."""
    out = {}
    for k, v in c.items():
        if k.startswith('X_'):
            x = v.numpy().astype(np.float64) * old.scale_ + old.mean_
            v = torch.tensor(new.transform(x), dtype=torch.float32)
        out[k] = v
    return out


@lru_cache(maxsize=1)
def load_data():
    """DATA TRAINING: hanya state terpilih (FL_STATES); kosong = semua state.
    Scaler di-fit hanya dari data train state terpilih."""
    clients, scaler, names = _load_raw()
    wanted = _wanted_states()
    if wanted is None:
        return clients, scaler, names

    hilang = wanted - set(clients)
    if hilang:
        print(f"[peringatan] state diminta tapi tidak ada/kosong: {sorted(hilang)}")
    pilih = [s for s in sorted(wanted) if s in clients]
    if not pilih:
        raise ValueError(f"Tidak ada state terpilih yang valid: {sorted(wanted)}")

    new_scaler = StandardScaler()
    for s in pilih:
        x = (clients[s]['X_train_t'].numpy().astype(np.float64)
             * scaler.scale_ + scaler.mean_)
        new_scaler.partial_fit(x)

    out = {s: _rescale_client(clients[s], scaler, new_scaler) for s in pilih}
    return out, new_scaler, names


@lru_cache(maxsize=1)
def load_eval_data():
    """DATA EVALUASI: SEMUA client (val & test), di-scale dengan scaler milik
    client training (tanpa kebocoran dari client yang tidak ikut latih).
    Client yang ikut latih memakai tensor yang sama (hemat memori)."""
    raw_clients, raw_scaler, names = _load_raw()
    if _wanted_states() is None:
        return raw_clients, raw_scaler, names

    train_clients, new_scaler, _ = load_data()
    out = {}
    for s in sorted(raw_clients):
        out[s] = (train_clients[s] if s in train_clients
                  else _rescale_client(raw_clients[s], raw_scaler, new_scaler))
    return out, new_scaler, names


def load_eval_clients():
    return load_eval_data()[0]


def clear_cache():
    for f in (_load_raw, load_data, load_eval_data):
        f.cache_clear()


def load_all():
    """Client TRAINING (dipakai Flower; urutan = partition-id)."""
    return load_data()[0]


def state_list():
    """Urutan state training = partition-id pada simulasi."""
    return list(load_all().keys())


def get_in_dim():
    return next(iter(load_all().values()))['X_train_t'].shape[1]


def get_xy(st, split, clients=None):
    c = (clients or load_all())[st]
    return c[f'X_{split}_t'], c[f'y_{split}_t']


def get_weights(model) -> NDArrays:
    """Semua parameter (theta + phi) sebagai list numpy array."""
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_weights(model, weights: NDArrays):
    sd = model.state_dict()
    if len(weights) != len(sd):
        raise ValueError(f"Jumlah tensor tidak cocok: server {len(weights)} "
                         f"vs model client {len(sd)}")
    for (k, v), w in zip(sd.items(), weights):
        if tuple(v.shape) != tuple(np.shape(w)):
            raise ValueError(
                f"Bentuk '{k}' tidak cocok: server {tuple(np.shape(w))} vs model "
                f"client {tuple(v.shape)}. Biasanya CLIENT_DIR berbeda antara "
                f"server dan worker (jumlah fitur beda). Cek FL_CLIENT_DIR / "
                f"config.CLIENT_DIR.")
    new = {k: torch.as_tensor(np.array(w)).to(dtype=sd[k].dtype)
           for k, w in zip(sd.keys(), weights)}
    model.load_state_dict(new)


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def nbytes(weights: NDArrays) -> int:
    return int(sum(w.nbytes for w in weights))


def class_weights(y):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos > 0 and n_neg > 0:
        return torch.tensor([len(y) / (2 * n_neg), len(y) / (2 * n_pos)],
                            dtype=torch.float32)
    return torch.ones(2)


def log_error(where):
    """Tulis traceback lengkap ke file."""
    try:
        config.ensure_out_dir()
        with open(config.OUT_DIR / 'client_error.log', 'a', encoding='utf-8') as f:
            f.write(f"\n===== {where} =====\n{traceback.format_exc()}\n")
    except Exception:
        pass


def _finite(weights):
    return all(np.isfinite(a).all() for a in weights if a.dtype.kind == 'f')


# ============================================================
# 2. EVALUASI (acc, prec, rec, f1, auc, prauc)
#    Default: SEMUA client (val & test), walau training hanya sebagian.
# ============================================================
@torch.no_grad()
def evaluate_model(model, X, y, criterion=None):
    model.eval()
    X, y = X.float().to(config.DEVICE), y.long().to(config.DEVICE)
    logits = model(X)
    prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    pred = logits.argmax(dim=1).cpu().numpy()
    y_np = y.cpu().numpy()

    finite = bool(np.isfinite(prob).all())
    two_class = len(np.unique(y_np)) > 1
    out = {
        'acc':   accuracy_score(y_np, pred),
        'prec':  precision_score(y_np, pred, zero_division=0),
        'rec':   recall_score(y_np, pred, zero_division=0),
        'f1':    f1_score(y_np, pred, zero_division=0),
        'auc':   (roc_auc_score(y_np, prob)
                  if two_class and finite else float('nan')),
        # PR-AUC = Average Precision (butuh minimal satu sampel positif)
        'prauc': (average_precision_score(y_np, prob)
                  if y_np.sum() > 0 and finite else float('nan')),
    }
    if criterion is not None:
        out['loss'] = criterion(logits, y).item()
    return out


def weighted_metrics(results):
    """results: list (n_examples, dict_metrik) -> rata-rata berbobot per metrik."""
    out = {}
    for k in METRIK:
        pairs = [(n, m[k]) for n, m in results
                 if k in m and not np.isnan(m[k])]
        tot = sum(n for n, _ in pairs)
        out[k] = sum(n * v for n, v in pairs) / tot if tot else float('nan')
    return out


def evaluate_on_clients(model, split='test', clients=None):
    """Evaluasi per client pada SEMUA client evaluasi (bukan hanya yang ikut latih)."""
    clients = clients or load_eval_clients()
    model = model.to(config.DEVICE)
    latih = set(load_all().keys())
    rows = []
    for st, c in clients.items():
        y = c[f'y_{split}_t']
        if len(y) == 0:
            continue
        m = evaluate_model(model, c[f'X_{split}_t'], y)
        rows.append({'state': st, 'ikut_latih': st in latih,
                     f'n_{split}': len(y), 'n_pos': int(y.sum()), **m})
    return pd.DataFrame(rows).set_index('state')


def summarize(df, n_col='n_test'):
    """Ringkasan makro, berbobot, dan std antar client."""
    bobot = df[n_col] / df[n_col].sum()
    ringkas = pd.DataFrame({
        'Rata-rata (makro)':  df[METRIK].mean(),
        'Rata-rata berbobot': df[METRIK].mul(bobot, axis=0).sum(skipna=True),
        'Std antar client':   df[METRIK].std(),
        'Min':                df[METRIK].min(),
        'Max':                df[METRIK].max(),
    })
    return ringkas.rename(index=NAMA_METRIK)


def evaluate_pooled(model, split='test', clients=None, states=None):
    """Gabungkan data client (semua, atau hanya `states`) jadi satu lalu evaluasi."""
    clients = clients or load_eval_clients()
    model = model.to(config.DEVICE)
    ks = [s for s, c in clients.items()
          if (states is None or s in states) and len(c[f'y_{split}_t']) > 0]
    if not ks:
        return {k: float('nan') for k in METRIK}
    X = torch.cat([clients[s][f'X_{split}_t'] for s in ks])
    y = torch.cat([clients[s][f'y_{split}_t'] for s in ks])
    return evaluate_model(model, X, y)


def load_final_model(path=None):
    """Bangun model dari bobot global terakhir (final_params.npz)."""
    path = Path(path) if path else config.OUT_DIR / 'final_params.npz'
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ada. Jalankan training dulu.")
    z = np.load(path)
    keys = sorted(z.files, key=lambda s: int(s.split('_')[1]))   # arr_0, arr_1, ...
    weights = [z[k] for k in keys]
    model = build_model(get_in_dim(), seed=get_seed())
    set_weights(model, weights)
    return model.to(config.DEVICE)


def full_report(model=None, splits=('val', 'test'), save=True, verbose=True):
    """
    Evaluasi model global pada SEMUA client (mis. 44) untuk tiap split,
    walau training hanya memakai sebagian (mis. 10).

    res[split] = {'per_client', 'ringkas', 'pooled',
                  'pooled_group': {'latih': {...}, 'non_latih': {...}},
                  'group_makro': DataFrame}
    Output CSV (bila save=True) di OUT_DIR:
      eval_{split}_per_client.csv, eval_{split}_ringkasan.csv,
      eval_{split}_per_grup.csv, eval_val_vs_test.csv
    """
    if model is None:
        model = load_final_model()
    config.ensure_out_dir()
    clients = load_eval_clients()
    latih = set(load_all().keys())
    non_latih = set(clients) - latih
    res = {}
    for sp in splits:
        per = evaluate_on_clients(model, sp, clients)
        ring = summarize(per, n_col=f'n_{sp}')
        pooled = evaluate_pooled(model, sp, clients)
        pooled_g = {
            'latih':     evaluate_pooled(model, sp, clients, latih),
            'non_latih': evaluate_pooled(model, sp, clients, non_latih),
        }
        group = per.groupby('ikut_latih')[METRIK].mean()
        group.index = group.index.map({True: 'client ikut latih',
                                       False: 'client TIDAK ikut latih'})
        res[sp] = {'per_client': per, 'ringkas': ring, 'pooled': pooled,
                   'pooled_group': pooled_g, 'group_makro': group}

        if save:
            per.to_csv(config.OUT_DIR / f'eval_{sp}_per_client.csv')
            ring.to_csv(config.OUT_DIR / f'eval_{sp}_ringkasan.csv')
            group.to_csv(config.OUT_DIR / f'eval_{sp}_per_grup.csv')
        if verbose:
            print(f"\n{'=' * 70}\nEVALUASI {sp.upper()} - per client "
                  f"({len(per)} client; {len(latih)} ikut latih)\n{'=' * 70}")
            print(per.round(4).to_string())
            print(f"\n--- Ringkasan {sp.upper()} (semua client) ---")
            print(ring.round(4).to_string())
            print(f"\n--- Makro per grup {sp.upper()} ---")
            print(group.round(4).to_string())
            print(f"\n--- Pooled {sp.upper()} (semua client digabung) ---")
            print("  ".join(f"{NAMA_METRIK[k]}={pooled[k]:.4f}" for k in METRIK))
            for g, p in pooled_g.items():
                print(f"  [{g:<9}] " + "  ".join(
                    f"{NAMA_METRIK[k]}={p[k]:.4f}" for k in METRIK))

    # tabel perbandingan val vs test
    cols = {}
    for sp in splits:
        cols[f'{sp}_pooled'] = pd.Series({k: res[sp]['pooled'][k] for k in METRIK})
        cols[f'{sp}_makro'] = res[sp]['per_client'][METRIK].mean()
        cols[f'{sp}_berbobot'] = res[sp]['ringkas']['Rata-rata berbobot'].rename(
            index={v: k for k, v in NAMA_METRIK.items()})
    cmp_df = pd.DataFrame(cols).rename(index=NAMA_METRIK)
    res['val_vs_test'] = cmp_df
    if save:
        cmp_df.to_csv(config.OUT_DIR / 'eval_val_vs_test.csv')
    if verbose:
        print(f"\n{'=' * 70}\nPERBANDINGAN VAL vs TEST\n{'=' * 70}")
        print(cmp_df.round(4).to_string())
    return res


# ============================================================
# 3. CLIENT (FedAvg / FedProx; FedOpt, FedPairwise & FedSim = sisi server saja)
# ============================================================
class FlowerClient(NumPyClient):
    def __init__(self, st, data):
        self.st = st
        torch.set_num_threads(1)          # 1 core per actor, hindari oversubscription
        dev = config.DEVICE
        self.X_tr, self.y_tr = data['X_train_t'].float(), data['y_train_t'].long()
        self.X_va, self.y_va = data['X_val_t'].float(), data['y_val_t'].long()
        self.model = build_model(in_dim=self.X_tr.shape[1],
                                 seed=get_seed()).to(dev)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights(self.y_tr).to(dev))

    def get_parameters(self, cfg):
        return get_weights(self.model)

    def fit(self, parameters, cfg):
        try:
            dev = config.DEVICE
            algo = cfg.get('algo', 'fedavg')
            mu = float(cfg.get('mu', 0.0))

            set_weights(self.model, parameters)
            params = trainable_params(self.model)
            glob = [p.detach().clone() for p in params]
            opt = torch.optim.Adam(params, lr=config.LR)

            seed = int(cfg.get('seed', get_seed()))
            rnd = int(cfg.get('round', 0))
            sts = state_list()
            pid = sts.index(self.st) if self.st in sts else 0
            gen = torch.Generator().manual_seed(seed * 100003 + rnd * 1009 + pid)

            n_tr = len(self.y_tr)
            loader = DataLoader(TensorDataset(self.X_tr, self.y_tr),
                                batch_size=config.BATCH_SIZE, shuffle=True,
                                generator=gen,
                                drop_last=(n_tr % config.BATCH_SIZE == 1))

            self.model.train()
            total, n = 0.0, 0
            for _ in range(config.LOCAL_EPOCHS):
                for xb, yb in loader:
                    xb, yb = xb.to(dev), yb.to(dev)
                    opt.zero_grad()
                    ce = self.criterion(self.model(xb), yb)
                    loss = ce
                    if algo == 'fedprox' and mu > 0:
                        prox = sum(((p - g) ** 2).sum()
                                   for p, g in zip(params, glob))
                        loss = ce + 0.5 * mu * prox
                    loss.backward()
                    opt.step()
                    total += ce.item() * len(xb)
                    n += len(xb)

            return get_weights(self.model), n_tr, {
                'train_loss': total / max(n, 1),
                'client_name': self.st,
            }
        except Exception:
            log_error(f"fit state={self.st}")
            raise

    def evaluate(self, parameters, cfg):
        """Evaluasi pada data VALIDASI client training; kirim semua 6 metrik ke server."""
        try:
            set_weights(self.model, parameters)
            m = evaluate_model(self.model, self.X_va, self.y_va, self.criterion)
            loss = m.pop('loss')
            return float(loss), len(self.y_va), {k: float(v) for k, v in m.items()}
        except Exception:
            log_error(f"evaluate state={self.st}")
            raise


def client_fn(context: Context) -> Client:
    pid = int(context.node_config['partition-id'])
    st = state_list()[pid]
    return FlowerClient(st, load_all()[st]).to_client()


def make_client_app() -> ClientApp:
    return ClientApp(client_fn=client_fn)


# ============================================================
# 4. STRATEGY: mixin pencatat bandwidth + metrik val per ronde
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
            print(f"[ronde {server_round:02d}] bobot pairwise: "
                  + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, w)))

        # Log cluster FedSim (hanya ada bila strategy-nya FedSim)
        labels = getattr(self, 'last_labels', None)
        if labels is not None:
            names = [r.metrics.get('client_name', f'client_{i}')
                     for i, (_, r) in enumerate(results)]
            print(f"[ronde {server_round:02d}] cluster fedsim: "
                  + ", ".join(f"{n}->g{int(l)}" for n, l in zip(names, labels)))
            pd.DataFrame({'client': names, 'cluster': labels}).to_csv(
                config.OUT_DIR / f'clusters_r{server_round:02d}.csv', index=False)
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


def make_strategy(algo, init, n_clients, model):
    """Bangun strategy sesuai nama algoritma."""
    algo = algo.lower() if algo not in ALGOS else algo
    # nama varian memuat huruf besar (mis. 'T'), jadi cocokkan tanpa peduli huruf besar/kecil
    if algo not in ALGOS:
        alias = {a.lower(): a for a in ALGOS}
        algo = alias.get(algo.lower(), algo)
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

    if algo in ('fedavg', 'fedprox'):
        cls = type('FedAvgBandwidth', (BandwidthMixin, FedAvg), {})
        return cls(**common)

    if algo == 'fedavgm':
        cls = type('FedAvgMBandwidth', (BandwidthMixin, FedAvgM), {})
        return cls(server_learning_rate=float(_cfg('SERVER_LR', 1.0)),
                   server_momentum=float(_cfg('SERVER_MOMENTUM', 0.9)), **common)

    if algo in PW_VARIANTS:
        if FedCosinePairwise is None:
            raise ImportError("aggregation4.FedCosinePairwise tidak tersedia; "
                              "pastikan aggregation4.py ada di folder yang sama.")
        temp, d_inf = PW_VARIANTS[algo]
        cls = type('FedPairwiseBandwidth', (BandwidthMixin, FedCosinePairwise), {})
        pw = dict(temperature=float(temp),
                  min_weight=float(_cfg('PAIRWISE_MIN_WEIGHT', 0.0)),
                  similarity_agg=_cfg('PAIRWISE_SIM_AGG', 'mean'),
                  data_influence=float(d_inf))
        if 'use_delta' in inspect.signature(FedCosinePairwise.__init__).parameters:
            pw['use_delta'] = bool(_cfg('PAIRWISE_USE_DELTA', True))
        else:
            print("[peringatan] aggregation4.py versi lama: use_delta diabaikan "
                  "(similarity dihitung pada bobot penuh)")
        return cls(**pw, **common)

    if algo in FS_VARIANTS:
        if FedSim is None:
            raise ImportError("fedsim_strategy.FedSim tidak tersedia; "
                              "pastikan fedsim_strategy.py ada di folder yang sama.")
        n_groups, g_weighting = FS_VARIANTS[algo]
        cls = type('FedSimBandwidth', (BandwidthMixin, FedSim), {})
        return cls(num_groups=int(n_groups),
                   pca_variance=float(_cfg('FEDSIM_PCA_VAR', 0.95)),
                   group_weighting=g_weighting,
                   seed=get_seed(),
                   **common)

    base = {'fedadam': FedAdam, 'fedyogi': FedYogi, 'fedadagrad': FedAdagrad}[algo]
    cls = type(f'{base.__name__}Bandwidth', (BandwidthMixin, base), {})
    return cls(eta=float(_cfg('SERVER_LR', 0.01)),
               eta_l=float(config.LR),
               beta_1=float(_cfg('BETA_1', 0.9)),
               beta_2=float(_cfg('BETA_2', 0.99)),
               tau=float(_cfg('TAU', 1e-3)), **common)


# ============================================================
# 5. SERVER
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


# ============================================================
# 6. BANDWIDTH
# ============================================================
def model_size_report(in_dim=None, verbose=True):
    in_dim = in_dim or get_in_dim()
    m = build_model(in_dim, seed=get_seed())
    n_theta = sum(p.numel() for p in m.theta.parameters())
    n_phi = sum(p.numel() for p in m.phi.parameters())
    n_total = n_theta + n_phi
    rep = {'n_theta': n_theta, 'n_phi': n_phi, 'n_total': n_total,
           'bytes_per_transfer': n_total * 4}
    if verbose:
        print("=== Ukuran model ===")
        print(f"theta : {n_theta:>8,} param ({n_theta * 4 / 1e3:.1f} KB)")
        print(f"phi   : {n_phi:>8,} param ({n_phi * 4 / 1e3:.1f} KB)")
        print(f"FULL  : {n_total:>8,} param ({n_total * 4 / 1e3:.1f} KB) <- dikirim tiap arah")
    return rep


def bandwidth_report(verbose=True):
    ronde = pd.read_csv(config.OUT_DIR / 'ronde_log.csv')
    size = model_size_report(verbose=False)
    n_clients = len(load_all())

    down, up = ronde['down_MB'].sum(), ronde['up_MB'].sum()
    total = down + up
    theta_only = total * size['n_theta'] / size['n_total']

    rep = {
        'total_down_MB': down, 'total_up_MB': up, 'total_MB': total,
        'mean_per_round_MB': ronde['total_MB'].mean(),
        'mean_per_client_MB': total / n_clients,
        'theta_only_MB': theta_only,
        'saving_theta_only_%': 100 * (1 - size['n_theta'] / size['n_total']),
    }
    if verbose:
        print("=== Bandwidth komunikasi (FULL parameter) ===")
        print(f"Total downlink (server->client): {down:,.3f} MB")
        print(f"Total uplink   (client->server): {up:,.3f} MB")
        print(f"TOTAL                          : {total:,.3f} MB")
        print(f"Rata-rata per ronde            : {rep['mean_per_round_MB']:.3f} MB")
        print(f"Rata-rata per client (total)   : {rep['mean_per_client_MB']:.3f} MB")
        print(f"[Pembanding] bila hanya theta  : {theta_only:,.3f} MB "
              f"(hemat {rep['saving_theta_only_%']:.2f}%)")
    return rep, ronde


# ============================================================
# 7. PAKET LENGKAP SETELAH TRAINING (dipanggil dari fl_basic / main.py)
# ============================================================
def final_evaluation(verbose=True):
    """Panggil SETELAH simulasi selesai: evaluasi val + test semua client + bandwidth."""
    res = full_report(verbose=verbose)
    try:
        bandwidth_report(verbose=verbose)
    except Exception as e:
        print(f"[peringatan] bandwidth_report dilewati: {e!r}")
    return res