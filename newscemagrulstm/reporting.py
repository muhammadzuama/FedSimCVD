"""Laporan bandwidth + paket evaluasi akhir (dipanggil dari main.py setelah training)."""
import pandas as pd

import config
from data_utils import get_in_dim, load_all
from evaluation import full_report
from model import build_model
from utils import get_seed


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


def final_evaluation(verbose=True):
    """Panggil SETELAH simulasi selesai: evaluasi val + test semua client + bandwidth."""
    res = full_report(verbose=verbose)
    try:
        bandwidth_report(verbose=verbose)
    except Exception as e:
        print(f"[peringatan] bandwidth_report dilewati: {e!r}")
    return res