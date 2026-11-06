"""Helper kecil tanpa state: config, seed, bobot model, logging error.
Modul ini TIDAK boleh meng-import modul proyek lain selain config."""
import os
import traceback

import numpy as np
import torch
from flwr.common import NDArrays

import config


def _cfg(name, default):
    return getattr(config, name, default)


def get_seed():
    """Seed aktif: FL_SEED (diset main.py) menang atas config.SEED."""
    v = os.environ.get('FL_SEED')
    return int(v) if v not in (None, '') else int(config.SEED)


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