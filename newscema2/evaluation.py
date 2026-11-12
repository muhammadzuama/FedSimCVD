"""Evaluasi (acc, prec, rec, f1, auc, prauc).
Default: SEMUA client (val & test), walau training hanya sebagian.
Bisa dijalankan tanpa Flower."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

import config
from data_utils import get_in_dim, load_all, load_eval_clients
from model import build_model
from utils import get_seed, set_weights

# Metrik lengkap (tidak bergantung pada config.METRIK yang mungkin hanya f1/auc)
METRIK = ['acc', 'prec', 'rec', 'f1', 'auc', 'prauc']
NAMA_METRIK = {
    'acc': 'Accuracy', 'prec': 'Precision', 'rec': 'Recall',
    'f1': 'F1-score', 'auc': 'ROC-AUC', 'prauc': 'PR-AUC',
}


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