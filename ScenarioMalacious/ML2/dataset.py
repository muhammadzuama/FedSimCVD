import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

TARGET = "_michd"
DROP_COLS = ["iyear", "_state", "cvdinfr4", "cvdcrhd4"]

# False = scaler hanya dari data asli (data sintetis cuma di-transform).
# True  = data sintetis ikut dipakai untuk fit scaler.
SCALE_WITH_SYNTH = False


def _to_xy(df):
    feats = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
    X = feats.to_numpy(dtype=np.float64)
    y = (df[TARGET] == 1).to_numpy(dtype=np.int64)
    return X, y, list(feats.columns)


def _parse_extra_clients():
    """FL_EXTRA_CLIENTS='1001=C:\\path\\folder;1002=...' -> {1001: Path}."""
    spec = os.environ.get("FL_EXTRA_CLIENTS", "").strip()
    out = {}
    for item in filter(None, spec.split(";")):
        sid, path = item.split("=", 1)
        out[int(sid)] = Path(path)
    return out


def prepare_client_data(client_dir):
    """
    Client asli : train + val + test
    Client extra (sintetis, dari FL_EXTRA_CLIENTS): train + val saja
                  (tidak punya kunci X_test_t / y_test_t)
    """
    raw, extra_ids = {}, set()

    # 1a. Client asli
    for folder in sorted(Path(client_dir).glob("state_*")):
        state = int(folder.name.split("_")[1])
        dfs = {s: pd.read_csv(folder / f"{s}.csv") for s in ("train", "val", "test")}
        if any(len(d) == 0 for d in dfs.values()):
            print(f"State {state} dilewati (ada split yang kosong).")
            continue
        raw[state] = {s: _to_xy(d) for s, d in dfs.items()}

    # 1b. Client sintetis (train + val saja)
    for state, folder in _parse_extra_clients().items():
        if state in raw:
            raise ValueError(f"ID {state} bentrok dengan client asli")
        dfs = {s: pd.read_csv(folder / f"{s}.csv") for s in ("train", "val")}
        if any(len(d) == 0 for d in dfs.values()):
            print(f"Client extra {state} dilewati (split kosong).")
            continue
        raw[state] = {s: _to_xy(d) for s, d in dfs.items()}
        extra_ids.add(state)

    if not raw:
        raise ValueError(f"Tidak ada data client yang valid di {client_dir}")

    # 2. Validasi kolom identik
    feature_names = next(iter(raw.values()))["train"][2]
    for state, parts in raw.items():
        for split, (_, _, cols) in parts.items():
            if cols != feature_names:
                raise ValueError(f"Kolom fitur beda di state {state}/{split}")

    # 3. Scaler
    scaler = StandardScaler()
    for state, parts in raw.items():
        if state in extra_ids and not SCALE_WITH_SYNTH:
            continue
        scaler.partial_fit(parts["train"][0])

    # 4. Tensor
    clients = {}
    for state, parts in raw.items():
        clients[state] = {}
        for split, (X, y, _) in parts.items():
            clients[state][f"X_{split}_t"] = torch.tensor(scaler.transform(X), dtype=torch.float32)
            clients[state][f"y_{split}_t"] = torch.tensor(y, dtype=torch.long)

    return clients, scaler, feature_names