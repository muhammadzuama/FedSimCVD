from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

TARGET = "_michd"
# Kolom yang dibuang dari fitur (diabaikan kalau tidak ada)
DROP_COLS = ["iyear", "_state", "cvdinfr4", "cvdcrhd4"]

def _to_xy(df):
    feats = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
    X = feats.to_numpy(dtype=np.float64)
    y = (df[TARGET] == 1).to_numpy(dtype=np.int64)   # 1 -> 1, selain itu -> 0
    return X, y, list(feats.columns)

def prepare_client_data(client_dir):
    """
    Baca data tiap state, scaling (fit hanya dari train), lalu ubah ke tensor.

    Returns:
        clients       : {state: {X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t}}
        scaler        : StandardScaler global (dari gabungan train semua client)
        feature_names : list nama fitur, urutan sama dengan kolom di tensor X
    """
    # 1. Baca data tiap client
    raw = {}
    for folder in sorted(Path(client_dir).glob("state_*")):
        state = int(folder.name.split("_")[1])
        dfs = {s: pd.read_csv(folder / f"{s}.csv") for s in ("train", "val", "test")}
        if any(len(d) == 0 for d in dfs.values()):
            print(f"State {state} dilewati (ada split yang kosong).")
            continue
        raw[state] = {s: _to_xy(d) for s, d in dfs.items()}

    if not raw:
        raise ValueError(f"Tidak ada data client yang valid di {client_dir}")

    # 2. Validasi: semua client & split harus punya kolom identik
    feature_names = next(iter(raw.values()))["train"][2]
    for state, parts in raw.items():
        for split, (_, _, cols) in parts.items():
            if cols != feature_names:
                raise ValueError(f"Kolom fitur beda di state {state}/{split}")

    # 3. Scaling: fit hanya dari train semua client
    scaler = StandardScaler()
    for parts in raw.values():
        scaler.partial_fit(parts["train"][0])

    # 4. Ubah ke tensor
    clients = {}
    for state, parts in raw.items():
        clients[state] = {}
        for split, (X, y, _) in parts.items():
            clients[state][f"X_{split}_t"] = torch.tensor(scaler.transform(X), dtype=torch.float32)
            clients[state][f"y_{split}_t"] = torch.tensor(y, dtype=torch.long)

    return clients, scaler, feature_names