"""Loading & scaling data.
    - load_data()      : data TRAINING  (hanya state di FL_STATES)
    - load_eval_data() : data EVALUASI  (SEMUA client, val & test)
Semua fungsi ber-cache (lru_cache) ada di sini supaya clear_cache() konsisten."""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import config
from dataset import prepare_client_data


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