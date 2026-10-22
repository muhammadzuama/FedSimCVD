"""
fl_runner.py  (gabungan + metrik kurva diagnostik & evaluasi ambang batas)

Federated learning dengan model terpisah:
    Adapter (lokal) -> theta (dibagi ke server) -> phi (lokal)

mode        : "sim" (FedSim) | "alt" (FedAlt)
aggregation : "fedavg" | "fedprox" | "pairwise"
    - fedavg   : rata-rata berbobot theta, mu = proximal_mu (default 0)
    - fedprox  : rata-rata berbobot theta + suku proximal (mu = fedprox_mu)
    - pairwise : agregasi pairwise-cosine (aggregation4)

Fitur gabungan:
    - class weight, threshold, kolom "n" per-client, n_client_fit per ronde
    - seed deterministik (init adapter/phi, urutan batch)
    - theta_param_only: global_theta sejajar dengan theta.parameters()
      (buffer BatchNorm dibuang), aman untuk FedProx
    - pilih ronde terbaik via validasi (select_by) -> test_best
    - fine-tune opsional adapter+phi sebelum evaluasi (eval_finetune_epochs)
    - laporan lengkap (report_results) + komparasi antar agregasi & seed

Metrik kurva diagnostik & evaluasi ambang batas (BARU):
    - AUC-ROC  ("auc") : kemampuan diskriminasi di semua threshold.
    - AP       ("ap")  : Average Precision = luas di bawah Precision-Recall
                         curve; lebih informatif pada data tidak seimbang.
                         Baseline AP acak = prevalensi kelas positif.
    - Kurva ROC & PR per client + POOLED disimpan di history
      (test_curves / test_best_curves), bisa di-plot & diekspor ke CSV.
    - Threshold sweep (precision, recall/sensitivitas, spesifisitas, F1,
      Youden J per threshold) -> test_sweep / test_best_sweep.
    - Pemilihan threshold dari VALIDASI lalu dipakai di TEST (tanpa
      kebocoran): threshold_mode = "fixed" | "per_client" | "global",
      threshold_criterion = "f1" | "youden" | "gmean" | "sens_target".
"""
import csv
import os
import random
import shutil
import tempfile
import warnings
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from model import build_model

DEVICE = torch.device("cpu")

MB = 1024 ** 2
METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "auc", "ap")
VALID_MODES = ("sim", "alt")
VALID_AGGS = ("fedavg", "fedprox", "pairwise")
VALID_THR_MODES = ("fixed", "per_client", "global")
VALID_THR_CRITERIA = ("f1", "youden", "gmean", "sens_target")
POOLED = "__POOLED__"          # kunci kurva/sweep gabungan semua client
MAX_CURVE_POINTS = 200


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Wrapper: Adapter (lokal) -> theta (dibagi) -> phi (lokal)
# -----------------------------------------------------------------------------
class ClientModel(nn.Module):
    def __init__(self, n_features_raw: int, common_dim: int, n_classes: int):
        super().__init__()
        self.adapter = nn.Linear(n_features_raw, common_dim)
        self.base = build_model(common_dim, n_classes)  # punya .theta dan .phi
        if not hasattr(self.base, "theta") or not hasattr(self.base, "phi"):
            raise AttributeError(
                "build_model(...) harus mengembalikan model dengan atribut "
                "`.theta` (FeatureExtractor) dan `.phi` (ClassifierHead)."
            )

    @property
    def theta(self) -> nn.Module:
        return self.base.theta

    @property
    def phi(self) -> nn.Module:
        return self.base.phi

    def personal_parameters(self) -> List[nn.Parameter]:
        return list(self.adapter.parameters()) + list(self.phi.parameters())

    def shared_parameters(self) -> List[nn.Parameter]:
        return list(self.theta.parameters())

    def forward(self, x):
        z = torch.relu(self.adapter(x))
        feat = self.theta(z)
        return self.phi(feat)


# -----------------------------------------------------------------------------
# Utilitas parameter (hanya THETA yang dipertukarkan)
# -----------------------------------------------------------------------------
def get_theta_params(model: ClientModel) -> NDArrays:
    return [v.detach().cpu().numpy() for v in model.theta.state_dict().values()]


def set_theta_params(model: ClientModel, params: NDArrays) -> None:
    keys = model.theta.state_dict().keys()
    state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)})
    model.theta.load_state_dict(state, strict=True)


def theta_param_only(model: ClientModel, params: NDArrays) -> List[torch.Tensor]:
    """
    Ambil dari `params` (urutan state_dict) hanya entri yang merupakan
    nn.Parameter (buang buffer BatchNorm), supaya cocok berpasangan dengan
    model.theta.parameters() pada suku proximal.
    """
    pnames = {n for n, _ in model.theta.named_parameters()}
    names = list(model.theta.state_dict().keys())
    return [torch.tensor(p).to(DEVICE) for n, p in zip(names, params)
            if n in pnames]


def params_nbytes(params: NDArrays) -> int:
    return int(sum(np.asarray(p).nbytes for p in params))


def full_model_nbytes(model: nn.Module) -> int:
    """Ukuran SELURUH model (adapter + theta + phi) dalam byte."""
    return int(sum(v.numel() * v.element_size() for v in model.state_dict().values()))


def to_tensors(X, y):
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    return (torch.as_tensor(np.asarray(X), dtype=torch.float32),
            torch.as_tensor(np.asarray(y), dtype=torch.long))


def compute_class_weight(y: torch.Tensor, n_classes: int,
                         power: float = 0.5) -> Optional[torch.Tensor]:
    if power <= 0:
        return None
    counts = torch.bincount(y, minlength=n_classes).float()
    present = counts > 0
    if present.sum() < 2:
        return None
    w = torch.ones(n_classes)
    w[present] = (counts[present].sum() / (present.sum() * counts[present])) ** power
    w = w / ((w * counts).sum() / counts.sum())
    return w


def _nanmean(vals) -> float:
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


# -----------------------------------------------------------------------------
# Penyimpanan bagian LOKAL (adapter + phi) per client
# -----------------------------------------------------------------------------
def local_path(local_dir: str, cid: int) -> str:
    return os.path.join(local_dir, f"local_client_{cid}.pt")


def save_local(model: ClientModel, local_dir: str, cid: int,
               fit_count: int = 0) -> None:
    torch.save({
        "adapter": {k: v.cpu() for k, v in model.adapter.state_dict().items()},
        "phi": {k: v.cpu() for k, v in model.phi.state_dict().items()},
        "fit_count": int(fit_count),
    }, local_path(local_dir, cid))


def read_fit_count(local_dir: str, cid: int) -> int:
    p = local_path(local_dir, cid)
    if os.path.exists(p):
        return int(torch.load(p, map_location="cpu", weights_only=True)
                   .get("fit_count", 0))
    return 0


def load_local_if_exists(model: ClientModel, local_dir: str, cid: int) -> bool:
    p = local_path(local_dir, cid)
    if os.path.exists(p):
        blob = torch.load(p, map_location="cpu", weights_only=True)
        model.adapter.load_state_dict(blob["adapter"])
        model.phi.load_state_dict(blob["phi"])
        return True
    return False


# -----------------------------------------------------------------------------
# Train: FedSim dan FedAlt (proximal_mu > 0 => suku FedProx pada theta)
# -----------------------------------------------------------------------------
def _set_requires_grad(params, flag: bool) -> None:
    for p in params:
        p.requires_grad_(flag)


def _run_epochs(model, loader, opt, epochs, proximal_mu=0.0, global_theta=None,
                class_weight=None):
    cw = class_weight.to(DEVICE) if class_weight is not None else None
    crit = nn.CrossEntropyLoss(weight=cw)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            if proximal_mu > 0 and global_theta is not None:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.theta.parameters(), global_theta))
                loss = loss + (proximal_mu / 2) * prox
            loss.backward()
            opt.step()


def train_local_sim(model, loader, epochs, lr, proximal_mu=0.0, global_theta=None,
                    class_weight=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    _run_epochs(model, loader, opt, epochs, proximal_mu, global_theta, class_weight)


def train_local_alt(model, loader, epochs_v, epochs_u, lr,
                    proximal_mu=0.0, global_theta=None, class_weight=None):
    model.train()
    personal = model.personal_parameters()
    shared = model.shared_parameters()
    try:
        _set_requires_grad(shared, False)
        _set_requires_grad(personal, True)
        opt_v = torch.optim.Adam(personal, lr=lr)
        _run_epochs(model, loader, opt_v, epochs_v, class_weight=class_weight)

        _set_requires_grad(personal, False)
        _set_requires_grad(shared, True)
        opt_u = torch.optim.Adam(shared, lr=lr)
        _run_epochs(model, loader, opt_u, epochs_u, proximal_mu, global_theta,
                    class_weight)
    finally:
        _set_requires_grad(personal, True)
        _set_requires_grad(shared, True)


train_local = train_local_sim


def finetune_local_head(model, X, y, epochs, lr, batch_size, seed):
    """Fine-tune adapter+phi di atas theta global (theta beku). Tidak disimpan."""
    X, y = to_tensors(X, y)
    for p in model.theta.parameters():
        p.requires_grad_(False)
    model.train()
    model.theta.eval()  # jangan update running stats BatchNorm
    params = list(model.adapter.parameters()) + list(model.phi.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size,
                        shuffle=True, generator=g)
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()


# -----------------------------------------------------------------------------
# Agregasi FedAvg (theta saja) -- dipakai juga untuk FedProx
# -----------------------------------------------------------------------------
class FedAvgTheta(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        entries = [(parameters_to_ndarrays(fr.parameters), fr.num_examples)
                   for _, fr in results]
        total = float(sum(n for _, n in entries))

        aggregated = []
        for k in range(len(entries[0][0])):
            ref = entries[0][0][k]
            acc = sum((n / total) * p[k].astype(np.float64) for p, n in entries)
            if np.issubdtype(ref.dtype, np.integer):
                acc = np.rint(acc)
            aggregated.append(acc.astype(ref.dtype))

        metrics = {}
        if self.fit_metrics_aggregation_fn:
            metrics = self.fit_metrics_aggregation_fn(
                [(fr.num_examples, fr.metrics) for _, fr in results])
        return ndarrays_to_parameters(aggregated), metrics


# -----------------------------------------------------------------------------
# Metrik skalar: AUC-ROC & Average Precision (AP)
# -----------------------------------------------------------------------------
def _safe_auc(yt: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    """AUC-ROC. Biner: pakai prob kelas 1. Multikelas: OvR macro."""
    try:
        if n_classes <= 2:
            if len(np.unique(yt)) < 2:
                return float("nan")
            return float(roc_auc_score(yt, probs[:, 1]))
        if len(np.unique(yt)) < n_classes:
            return float("nan")
        return float(roc_auc_score(yt, probs, multi_class="ovr", average="macro",
                                   labels=list(range(n_classes))))
    except ValueError:
        return float("nan")


def _safe_ap(yt: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    """
    Average Precision (= luas Precision-Recall curve).
    Biner: AP kelas positif (1). Multikelas: macro-AP one-vs-rest atas kelas
    yang muncul di yt. NaN bila AP tidak terdefinisi (kelas tunggal).
    """
    try:
        if n_classes <= 2:
            if len(np.unique(yt)) < 2:
                return float("nan")
            return float(average_precision_score(yt, probs[:, 1]))
        present = [c for c in range(n_classes) if (yt == c).any()]
        if len(present) < 2:
            return float("nan")
        Y = label_binarize(yt, classes=list(range(n_classes)))
        return float(np.mean([average_precision_score(Y[:, c], probs[:, c])
                              for c in present]))
    except ValueError:
        return float("nan")


# -----------------------------------------------------------------------------
# Kurva diagnostik (ROC & PR) + threshold sweep + pemilihan threshold
# -----------------------------------------------------------------------------
def _downsample(x: np.ndarray, y: np.ndarray, k: int = MAX_CURVE_POINTS):
    """Kurangi jumlah titik kurva (pasangan x,y tetap sejajar)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) <= k:
        return x.tolist(), y.tolist()
    idx = np.unique(np.linspace(0, len(x) - 1, k).astype(int))
    return x[idx].tolist(), y[idx].tolist()


def compute_curves(yt: np.ndarray, p1: np.ndarray) -> Optional[dict]:
    """Kurva ROC dan Precision-Recall (biner). None bila kelas tunggal."""
    yt, p1 = np.asarray(yt), np.asarray(p1)
    if len(np.unique(yt)) < 2:
        return None
    fpr, tpr, _ = roc_curve(yt, p1)
    prec, rec, _ = precision_recall_curve(yt, p1)
    fpr_d, tpr_d = _downsample(fpr, tpr)
    rec_d, prec_d = _downsample(rec, prec)
    return {
        "fpr": fpr_d, "tpr": tpr_d,
        "recall": rec_d, "precision": prec_d,
        "auc": float(roc_auc_score(yt, p1)),
        "ap": float(average_precision_score(yt, p1)),
        "prevalence": float((yt == 1).mean()),
        "n": int(len(yt)),
    }


def threshold_sweep(yt: np.ndarray, p1: np.ndarray,
                    thresholds=None) -> List[dict]:
    """Metrik operasi pada grid threshold (biner, kelas positif = 1)."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    yt, p1 = np.asarray(yt), np.asarray(p1)
    P, N = (yt == 1), (yt != 1)
    rows = []
    for t in thresholds:
        pred = p1 >= t
        tp = int((pred & P).sum())
        fp = int((pred & N).sum())
        fn = int((~pred & P).sum())
        tn = int((~pred & N).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        den = 2 * tp + fp + fn
        f1 = (2 * tp / den) if den > 0 else 0.0
        youden = (rec + spec - 1.0) if not (np.isnan(rec) or np.isnan(spec)) \
            else float("nan")
        rows.append({"threshold": float(t), "precision": prec, "recall": rec,
                     "specificity": spec, "f1": f1, "youden": youden,
                     "pos_pred_rate": float(pred.mean())})
    return rows


def select_threshold(yt: np.ndarray, p1: np.ndarray, criterion: str = "f1",
                     target_sensitivity: float = 0.9,
                     default: float = 0.5) -> float:
    """
    Pilih threshold optimal dari data (dipanggil pada VALIDASI).
      f1          : maksimalkan F1 (dari PR curve)
      youden      : maksimalkan sensitivitas + spesifisitas - 1 (dari ROC)
      gmean       : maksimalkan sqrt(sensitivitas * spesifisitas)
      sens_target : threshold tertinggi dengan sensitivitas >= target_sensitivity
                    (cocok untuk skrining medis: minimalkan false negative)
    Bila validasi hanya punya satu kelas, kembalikan `default`.
    """
    yt, p1 = np.asarray(yt), np.asarray(p1)
    if len(np.unique(yt)) < 2:
        return float(default)
    if criterion == "f1":
        prec, rec, thr = precision_recall_curve(yt, p1)
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
        return float(thr[int(np.argmax(f1))])
    fpr, tpr, thr = roc_curve(yt, p1)
    thr = np.where(np.isfinite(thr), thr, 1.0)   # titik awal ROC bisa inf
    if criterion == "youden":
        idx = int(np.argmax(tpr - fpr))
    elif criterion == "gmean":
        idx = int(np.argmax(np.sqrt(tpr * (1.0 - fpr))))
    elif criterion == "sens_target":
        idx = int(np.argmax(tpr >= target_sensitivity))
    else:
        raise ValueError(f"criterion harus salah satu dari {VALID_THR_CRITERIA}")
    return float(np.clip(thr[idx], 0.0, 1.0))


# -----------------------------------------------------------------------------
# Evaluasi
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model, X, y, batch_size: int = 4096):
    """Return (loss rata-rata, y numpy, probabilitas softmax numpy)."""
    model.eval()
    X, y = to_tensors(X, y)
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss, probs_all = 0.0, []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(DEVICE)
        yb = y[i:i + batch_size].to(DEVICE)
        logits = model(xb)
        total_loss += crit(logits, yb).item()
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs_all)
    yt = y.numpy()
    return total_loss / max(len(yt), 1), yt, probs


def compute_metrics(yt: np.ndarray, probs: np.ndarray, n_classes: int,
                    threshold: float = 0.5) -> dict:
    if n_classes <= 2:
        pred = (probs[:, 1] >= threshold).astype(np.int64)
        avg = "binary"
    else:
        pred = probs.argmax(1)
        avg = "macro"
    out = {
        "accuracy": float((pred == yt).mean()),
        "precision": float(precision_score(yt, pred, average=avg, zero_division=0)),
        "recall": float(recall_score(yt, pred, average=avg, zero_division=0)),
        "f1": float(f1_score(yt, pred, average=avg, zero_division=0)),
        "auc": _safe_auc(yt, probs, n_classes),
        "ap": _safe_ap(yt, probs, n_classes),
        "n": int(len(yt)),
    }
    if n_classes <= 2:
        out["pos_pred_rate"] = float((pred == 1).mean())
        out["prevalence"] = float((yt == 1).mean())
        out["threshold"] = float(threshold)
    return out


def evaluate_model(model, X, y, n_classes: int = 2, batch_size: int = 4096,
                   threshold: float = 0.5):
    loss, yt, probs = predict_proba(model, X, y, batch_size)
    return loss, compute_metrics(yt, probs, n_classes, threshold)


def evaluate_personalized_per_client(
    theta_params: NDArrays,
    fed_data: Dict[str, dict],
    common_dim: int,
    n_classes: int,
    local_dir: str,
    split: str,
    threshold: float = 0.5,
    finetune_epochs: int = 0,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
    threshold_mode: str = "fixed",
    threshold_criterion: str = "f1",
    target_sensitivity: float = 0.9,
    tune_split: str = "val",
    collect_curves: bool = False,
):
    """
    theta global + (adapter, phi) tiap client -> evaluasi di data client itu.

    threshold_mode:
      "fixed"      : semua client pakai `threshold`.
      "per_client" : threshold dipilih per client dari split `tune_split`.
      "global"     : satu threshold dari gabungan `tune_split` semua client.
    Tuning HANYA aktif bila split != tune_split (hindari optimisme: split val
    selama ronde tetap memakai threshold tetap) dan untuk kasus biner.

    Return: (loss_rata2, mean_metrik, per_client, extras)
      extras = {"curves", "sweep", "thresholds"} (terisi bila collect_curves).
    """
    tuning = (threshold_mode != "fixed" and n_classes <= 2
              and split != tune_split)

    cache: "OrderedDict[str, dict]" = OrderedDict()
    for cid, (name, c) in enumerate(fed_data.items()):
        Xk, yk = c.get(f"X_{split}"), c.get(f"y_{split}")
        if Xk is None or yk is None:
            continue
        model = ClientModel(c["n_features"], common_dim, n_classes).to(DEVICE)
        set_theta_params(model, theta_params)
        if not load_local_if_exists(model, local_dir, cid):
            continue
        if finetune_epochs > 0:
            finetune_local_head(model, c["X_train"], c["y_train"],
                                finetune_epochs, lr, batch_size, seed + cid)
        loss, yt, probs = predict_proba(model, Xk, yk)
        entry = {"loss": loss, "yt": yt, "probs": probs, "tune": None}
        if tuning:
            Xv, yv = c.get(f"X_{tune_split}"), c.get(f"y_{tune_split}")
            if Xv is not None and yv is not None:
                _, yv_np, pv = predict_proba(model, Xv, yv)
                entry["tune"] = (yv_np, pv[:, 1])
        cache[name] = entry

    if not cache:
        return None, None, {}, {}

    # ---- tentukan threshold per client ----
    thr_by = {n: float(threshold) for n in cache}
    if tuning:
        if threshold_mode == "per_client":
            for n, e in cache.items():
                if e["tune"] is not None:
                    thr_by[n] = select_threshold(
                        e["tune"][0], e["tune"][1], threshold_criterion,
                        target_sensitivity, threshold)
        else:  # "global"
            tunes = [e["tune"] for e in cache.values() if e["tune"] is not None]
            if tunes:
                g = select_threshold(
                    np.concatenate([t[0] for t in tunes]),
                    np.concatenate([t[1] for t in tunes]),
                    threshold_criterion, target_sensitivity, threshold)
                thr_by = {n: g for n in cache}

    # ---- metrik per client ----
    losses, per_client = [], {}
    for n, e in cache.items():
        losses.append(e["loss"])
        per_client[n] = compute_metrics(e["yt"], e["probs"], n_classes, thr_by[n])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_m = {k: _nanmean(m[k] for m in per_client.values())
                  for k in METRIC_KEYS}

    # ---- kurva ROC/PR + sweep (biner saja) ----
    extras: dict = {}
    if collect_curves and n_classes <= 2:
        curves, sweeps, ys, ps = {}, {}, [], []
        for n, e in cache.items():
            p1 = e["probs"][:, 1]
            cv = compute_curves(e["yt"], p1)
            if cv is not None:
                curves[n] = cv
            sweeps[n] = threshold_sweep(e["yt"], p1)
            ys.append(e["yt"])
            ps.append(p1)
        yp, pp = np.concatenate(ys), np.concatenate(ps)
        cv = compute_curves(yp, pp)
        if cv is not None:
            curves[POOLED] = cv
        sweeps[POOLED] = threshold_sweep(yp, pp)
        extras = {"curves": curves, "sweep": sweeps, "thresholds": thr_by}

    return float(np.mean(losses)), mean_m, per_client, extras


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------
class FlowerClient(NumPyClient):
    def __init__(self, cid, name, X, y, n_features_raw, common_dim, n_classes,
                 local_epochs, batch_size, lr, local_dir, proximal_mu=0.0,
                 mode="sim", epochs_v=None, epochs_u=None,
                 class_weight_power=0.5, seed=0):
        if mode not in VALID_MODES:
            raise ValueError(f"mode harus salah satu dari {VALID_MODES}, dapat: {mode!r}")
        self.cid = cid
        self.name = name
        self.seed = seed
        torch.set_num_threads(1)
        set_seed(seed + cid)  # init adapter/phi deterministik per client
        self.X, self.y = to_tensors(X, y)
        self.n_classes = n_classes
        self.model = ClientModel(n_features_raw, common_dim, n_classes).to(DEVICE)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.local_dir = local_dir
        self.proximal_mu = proximal_mu
        self.mode = mode
        self.epochs_v = epochs_v if epochs_v is not None else local_epochs
        self.epochs_u = epochs_u if epochs_u is not None else local_epochs
        self.class_weight = compute_class_weight(self.y, n_classes,
                                                 class_weight_power)

    def _restore_local(self):
        if load_local_if_exists(self.model, self.local_dir, self.cid):
            self.model.to(DEVICE)

    def fit(self, parameters, config):
        bytes_down = params_nbytes(parameters)

        set_theta_params(self.model, parameters)        # theta <- global
        self._restore_local()                           # adapter+phi <- lokal
        global_theta = theta_param_only(self.model, parameters)

        fit_count = read_fit_count(self.local_dir, self.cid)
        g = torch.Generator().manual_seed(self.seed + self.cid * 10007 + fit_count)
        loader = DataLoader(TensorDataset(self.X, self.y),
                            batch_size=self.batch_size, shuffle=True, generator=g)

        if self.mode == "alt":
            train_local_alt(self.model, loader, self.epochs_v, self.epochs_u,
                            self.lr, self.proximal_mu, global_theta,
                            self.class_weight)
        else:
            train_local_sim(self.model, loader, self.local_epochs, self.lr,
                            self.proximal_mu, global_theta, self.class_weight)

        save_local(self.model, self.local_dir, self.cid, fit_count + 1)

        new_params = get_theta_params(self.model)
        bytes_up = params_nbytes(new_params)
        return new_params, len(self.X), {
            "bytes_down": float(bytes_down),
            "bytes_up": float(bytes_up),
        }

    def evaluate(self, parameters, config):
        # Tidak dipakai (fraction_evaluate=0); evaluasi dilakukan server.
        set_theta_params(self.model, parameters)
        self._restore_local()
        loss, m = evaluate_model(self.model, self.X, self.y,
                                 n_classes=self.n_classes)
        m = {k: (0.0 if (isinstance(v, float) and np.isnan(v)) else v)
             for k, v in m.items()}
        return loss, len(self.X), m


# -----------------------------------------------------------------------------
# Strategi agregasi
# -----------------------------------------------------------------------------
def build_strategy(aggregation: str, num_clients: int, init_params, evaluate_fn,
                   fit_metrics_agg, strat_cfg: dict):
    if aggregation in ("fedavg", "fedprox"):
        # FedProx = agregasi FedAvg; perbedaannya di suku proximal sisi client.
        return FedAvgTheta(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
            initial_parameters=init_params,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=fit_metrics_agg,
        )
    from aggregation4 import build_fedcosine_pairwise_strategy
    return build_fedcosine_pairwise_strategy(
        num_clients=num_clients,
        init_params=init_params,
        evaluate_fn=evaluate_fn,
        fit_metrics_agg=fit_metrics_agg,
        **strat_cfg,
    )


# -----------------------------------------------------------------------------
# Pelaporan
# -----------------------------------------------------------------------------
def per_client_table(per_client: Dict[str, dict],
                     client_label: str = "state") -> pd.DataFrame:
    rows = {name: {"n_test": m.get("n", np.nan),
                   "acc": m["accuracy"], "prec": m["precision"],
                   "rec": m["recall"], "f1": m["f1"], "auc": m["auc"],
                   "ap": m.get("ap", np.nan),
                   "thr": m.get("threshold", np.nan)}
            for name, m in per_client.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["n_test"] = df["n_test"].astype(int)
    df.index.name = client_label
    return df


def summarize_clients(df: pd.DataFrame):
    """Return (tabel ringkasan makro/berbobot/std, dict rata-rata berbobot)."""
    labels = {"acc": "Akurasi", "prec": "Presisi", "rec": "Recall",
              "f1": "F1", "auc": "AUC-ROC", "ap": "AP (PR-AUC)"}
    w = df["n_test"].to_numpy(dtype=float)
    rows, weighted = {}, {}
    for k, lab in labels.items():
        v = df[k].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        macro = float(v[ok].mean()) if ok.any() else float("nan")
        wavg = float(np.average(v[ok], weights=w[ok])) if ok.any() else float("nan")
        std = float(pd.Series(v[ok]).std()) if ok.sum() > 1 else float("nan")
        rows[lab] = {"Rata-rata (makro)": macro,
                     "Rata-rata berbobot": wavg,
                     "Std antar client": std}
        weighted[k] = wavg
    return pd.DataFrame(rows).T, weighted


def print_round_table(history: Dict) -> None:
    n = len(history["round"])
    df = pd.DataFrame({
        "round": history["round"],
        "val_loss": history["loss"],
        "acc": history["accuracy"],
        "prec": history["precision"],
        "rec": history["recall"],
        "f1": history["f1"],
        "auc": history["auc"],
        "ap": history["ap"],
        "n_client_fit": history["n_client_fit"][:n],
        "down_MB": [b / MB for b in history["bytes_down"][:n]],
        "up_MB": [b / MB for b in history["bytes_up"][:n]],
        "total_MB": [b / MB for b in history["bytes_total"][:n]],
    })
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_bandwidth_report(history: Dict) -> None:
    c = history["comm"]
    print("=== Bandwidth komunikasi (yang benar-benar dikirim: theta) ===")
    print(f"Total downlink (server->client): {c['total_down_mb']:.3f} MB")
    print(f"Total uplink   (client->server): {c['total_up_mb']:.3f} MB")
    print(f"TOTAL                          : {c['total_mb']:.3f} MB")
    print(f"Rata-rata per ronde            : {c['avg_per_round_mb']:.3f} MB")
    print(f"Rata-rata per client (total)   : {c['avg_per_client_mb']:.3f} MB")
    print(f"[Pembanding] bila FULL parameter: {c['full_total_mb']:.3f} MB "
          f"(hemat {c['saving_pct']:.2f}%)")


def _print_split_report(per_client: Dict, title: str, client_label: str):
    df = per_client_table(per_client, client_label)
    summ, weighted = summarize_clients(df)
    print(f"\n=== {title} ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(summ.to_string(float_format=lambda x: f"{x:.4f}"))
    print(weighted)
    return df, summ, weighted


def _print_curve_summary(history: Dict, prefix: str) -> None:
    """Ringkasan kurva gabungan (pooled) + threshold yang dipakai."""
    cv = (history.get(f"{prefix}_curves") or {}).get(POOLED)
    if cv:
        base = cv["prevalence"]
        print(f"[pooled] AUC-ROC={cv['auc']:.4f} | AP={cv['ap']:.4f} "
              f"(baseline acak AP = prevalensi = {base:.4f}) | n={cv['n']}")
    thr = history.get(f"{prefix}_thresholds")
    if thr and history.get("threshold_mode", "fixed") != "fixed":
        s = ", ".join(f"{n}={t:.3f}" for n, t in thr.items())
        print(f"[threshold {history['threshold_mode']}/"
              f"{history.get('threshold_criterion')}] {s}")


def report_results(history: Dict, client_label: str = "state") -> Dict:
    """Cetak seluruh laporan dan kembalikan dict ringkasan berbobot (ronde terakhir)."""
    if history["round"]:
        i = len(history["round"]) - 1
        print(f"[ronde {history['round'][i]}] val F1 {history['f1'][i]:.4f} | "
              f"AUC {history['auc'][i]:.4f} | AP {history['ap'][i]:.4f} | "
              f"bandwidth ronde {history['bytes_total'][i] / MB:.3f} MB")
    print_bandwidth_report(history)
    print_round_table(history)

    weighted = {}
    pc = history.get("test_per_client")
    if pc:
        df, summ, weighted = _print_split_report(
            pc, "TEST (ronde terakhir)", client_label)
        _print_curve_summary(history, "test")
        history["test_table"], history["test_summary"] = df, summ
        history["test_weighted"] = weighted

    pcb = history.get("test_best_per_client")
    if pcb:
        df, summ, wb = _print_split_report(
            pcb, f"TEST (ronde terbaik val = {history.get('best_round')})",
            client_label)
        _print_curve_summary(history, "test_best")
        history["test_best_table"], history["test_best_summary"] = df, summ
        history["test_best_weighted"] = wb
    return weighted


def print_threshold_sweep(history: Dict, client: str = POOLED,
                          split: str = "test_best_sweep") -> None:
    """Cetak tabel threshold sweep (default: gabungan semua client)."""
    rows = (history.get(split) or {}).get(client)
    if not rows:
        print(f"[sweep] tidak ada data untuk '{client}' di {split}.")
        return
    print(f"\n=== Threshold sweep: {client} ({split}) ===")
    print(pd.DataFrame(rows).to_string(index=False,
                                        float_format=lambda x: f"{x:.4f}"))


# -----------------------------------------------------------------------------
# Plot kurva diagnostik (matplotlib, opsional)
# -----------------------------------------------------------------------------
def plot_diagnostic_curves(history: Dict, out_path: str,
                           split: str = "test_best_curves",
                           show_clients: bool = True,
                           title: Optional[str] = None) -> str:
    """Gambar ROC (kiri) dan Precision-Recall (kanan): per client + pooled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = history.get(split)
    if not curves:
        print(f"[plot] tidak ada kurva di '{split}' (hanya kasus biner).")
        return ""
    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, cv in curves.items():
        if name == POOLED:
            kw = dict(color="black", lw=2.5, zorder=5)
            lab = "POOLED"
        elif not show_clients:
            continue
        else:
            kw = dict(lw=1.2, alpha=0.8)
            lab = str(name)
        ax_r.plot(cv["fpr"], cv["tpr"], label=f"{lab} (AUC={cv['auc']:.3f})", **kw)
        ax_p.plot(cv["recall"], cv["precision"],
                  label=f"{lab} (AP={cv['ap']:.3f})", **kw)
    ax_r.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax_r.set(xlabel="False Positive Rate (1 - spesifisitas)",
             ylabel="True Positive Rate (sensitivitas)", title="Kurva ROC",
             xlim=(0, 1), ylim=(0, 1.02))
    if POOLED in curves:
        ax_p.axhline(curves[POOLED]["prevalence"], ls="--", color="gray", lw=1,
                     label=f"baseline acak ({curves[POOLED]['prevalence']:.3f})")
    ax_p.set(xlabel="Recall", ylabel="Precision", title="Kurva Precision-Recall",
             xlim=(0, 1), ylim=(0, 1.02))
    for ax in (ax_r, ax_p):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] kurva ROC/PR tersimpan di: {out_path}")
    return out_path


def plot_comparison_curves(histories: Dict[str, Dict], out_path: str,
                           split: str = "test_best_curves") -> str:
    """Bandingkan kurva POOLED antar agregasi (ROC kiri, PR kanan)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(13, 5.2))
    prev = None
    plotted = False
    for agg, h in histories.items():
        cv = (h.get(split) or {}).get(POOLED)
        if not cv:
            continue
        plotted = True
        prev = cv["prevalence"]
        ax_r.plot(cv["fpr"], cv["tpr"], lw=1.8, label=f"{agg} (AUC={cv['auc']:.3f})")
        ax_p.plot(cv["recall"], cv["precision"], lw=1.8,
                  label=f"{agg} (AP={cv['ap']:.3f})")
    if not plotted:
        plt.close(fig)
        print(f"[plot] tidak ada kurva pooled di '{split}'.")
        return ""
    ax_r.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax_r.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
             title="ROC (pooled)", xlim=(0, 1), ylim=(0, 1.02))
    if prev is not None:
        ax_p.axhline(prev, ls="--", color="gray", lw=1,
                     label=f"baseline acak ({prev:.3f})")
    ax_p.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (pooled)",
             xlim=(0, 1), ylim=(0, 1.02))
    for ax in (ax_r, ax_p):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] komparasi kurva antar agregasi tersimpan di: {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_federated(
    fed_data: Dict[str, dict],
    name: str,
    common_dim: int,
    n_classes: int = 2,
    aggregation: str = "pairwise",     # "fedavg" | "fedprox" | "pairwise"
    mode: str = "sim",                 # "sim" | "alt"
    proximal_mu: float = 0.0,          # mu untuk fedavg/pairwise (default off)
    fedprox_mu: float = 0.01,          # mu khusus aggregation="fedprox"
    num_rounds: int = 10,
    local_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs_v: Optional[int] = None,
    epochs_u: Optional[int] = None,
    strategy_kwargs: Optional[dict] = None,
    class_weight_power: float = 0.5,
    threshold: float = 0.5,
    threshold_mode: str = "fixed",         # "fixed" | "per_client" | "global"
    threshold_criterion: str = "f1",       # "f1" | "youden" | "gmean" | "sens_target"
    target_sensitivity: float = 0.9,       # dipakai bila criterion="sens_target"
    seed: int = 42,
    select_by: str = "f1",             # "loss" | salah satu METRIC_KEYS (val)
    eval_finetune_epochs: int = 0,
    verbose_report: bool = True,
    client_label: str = "state",
    max_parallel: int = 4,             # jumlah worker/actor Ray paralel (turunkan jika RAM kecil)
    on_round: Optional[Callable] = None,  # callback(round, num_rounds, val_loss, val_metrics) utk progress bar
) -> Dict:
    if mode not in VALID_MODES:
        raise ValueError(f"mode harus salah satu dari {VALID_MODES}, dapat: {mode!r}")
    if aggregation not in VALID_AGGS:
        raise ValueError(f"aggregation harus salah satu dari {VALID_AGGS}, dapat: {aggregation!r}")
    if select_by != "loss" and select_by not in METRIC_KEYS:
        raise ValueError(f"select_by harus 'loss' atau salah satu dari {METRIC_KEYS}")
    if threshold_mode not in VALID_THR_MODES:
        raise ValueError(f"threshold_mode harus salah satu dari {VALID_THR_MODES}")
    if threshold_criterion not in VALID_THR_CRITERIA:
        raise ValueError(f"threshold_criterion harus salah satu dari {VALID_THR_CRITERIA}")
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity harus di (0, 1].")
    if threshold_mode != "fixed" and n_classes > 2:
        warnings.warn("Tuning threshold hanya untuk kasus biner; memakai 'fixed'.")
        threshold_mode = "fixed"
    for cname, c in fed_data.items():
        if "n_features" not in c:
            raise KeyError(f"fed_data['{cname}'] harus punya key 'n_features'.")
    if threshold_mode != "fixed":
        for cname, c in fed_data.items():
            if c.get("X_val") is None or c.get("y_val") is None:
                raise KeyError(f"threshold_mode='{threshold_mode}' butuh "
                               f"X_val/y_val pada fed_data['{cname}'].")

    if aggregation == "fedprox":
        if fedprox_mu <= 0:
            raise ValueError("fedprox membutuhkan fedprox_mu > 0.")
        mu = fedprox_mu
    else:
        mu = proximal_mu

    if aggregation == "pairwise":
        from aggregation4 import build_fedcosine_pairwise_strategy  # noqa: F401

    strat_cfg = dict(temperature=0.01, min_weight=0.0,
                     similarity_agg="mean", data_influence=0.3)
    if strategy_kwargs:
        strat_cfg.update(strategy_kwargs)

    num_clients = len(fed_data)
    client_names = list(fed_data.keys())
    root_dir = tempfile.mkdtemp(prefix="fl_")
    local_dir = os.path.join(root_dir, "local")
    best_dir = os.path.join(root_dir, "best")
    os.makedirs(local_dir)
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir)
    # Data tiap client disimpan ke disk. client_fn hanya memegang path, sehingga
    # Ray tidak mem-pickle seluruh fed_data (semua client) ke setiap actor/pesan.
    for _cid, (_cn, _c) in enumerate(fed_data.items()):
        _X, _y = to_tensors(_c["X_train"], _c["y_train"])
        torch.save({"X": _X, "y": _y}, os.path.join(data_dir, f"client_{_cid}.pt"))
    n_features_list = [c["n_features"] for c in fed_data.values()]
    tag = f"{name}|{mode}|{aggregation}"

    history: Dict = {k: [] for k in
                     ["round", "loss", *METRIC_KEYS,
                      "bytes_up", "bytes_down", "bytes_total", "cum_mb",
                      "n_client_fit", "val_per_client"]}
    history.update(mode=mode, aggregation=aggregation, seed=seed, mu=mu,
                   threshold_mode=threshold_mode,
                   threshold_criterion=threshold_criterion)
    last = {"params": None, "best_params": None, "best_round": None,
            "best_score": None}

    def client_fn(context: Context) -> Client:
        cid = int(context.node_config["partition-id"])
        cname = client_names[cid]
        blob = torch.load(os.path.join(data_dir, f"client_{cid}.pt"),
                          map_location="cpu", weights_only=True)
        return FlowerClient(
            cid, cname, blob["X"], blob["y"], n_features_list[cid],
            common_dim, n_classes, local_epochs, batch_size, lr,
            local_dir, mu,
            mode=mode, epochs_v=epochs_v, epochs_u=epochs_u,
            class_weight_power=class_weight_power, seed=seed,
        ).to_client()

    def fit_metrics_agg(results):
        up = sum(m.get("bytes_up", 0.0) for _, m in results)
        down = sum(m.get("bytes_down", 0.0) for _, m in results)
        prev = history["cum_mb"][-1] if history["cum_mb"] else 0.0
        history["n_client_fit"].append(len(results))
        history["bytes_up"].append(up)
        history["bytes_down"].append(down)
        history["bytes_total"].append(up + down)
        history["cum_mb"].append(prev + (up + down) / MB)
        return {"bytes_up": up, "bytes_down": down}

    def is_better(score, best):
        if np.isnan(score):
            return False
        if best is None:
            return True
        return score < best if select_by == "loss" else score > best

    def evaluate_fn(server_round, parameters, config):
        if server_round == 0:
            return None
        theta = parameters if isinstance(parameters, list) \
            else [np.asarray(a) for a in parameters_to_ndarrays(parameters)]
        # validasi selama ronde: threshold tetap (tidak ada tuning pada val)
        loss, m, per_client, _ = evaluate_personalized_per_client(
            theta, fed_data, common_dim, n_classes, local_dir, split="val",
            threshold=threshold, finetune_epochs=eval_finetune_epochs,
            lr=lr, batch_size=batch_size, seed=seed)
        if m is None:
            return None
        last["params"] = theta
        history["round"].append(server_round)
        history["loss"].append(loss)
        history["val_per_client"].append(per_client)
        for k in METRIC_KEYS:
            history[k].append(m[k])

        # snapshot ronde terbaik (theta + adapter/phi lokal pada ronde itu)
        score = loss if select_by == "loss" else m[select_by]
        if is_better(score, last["best_score"]):
            last["best_score"] = score
            last["best_round"] = server_round
            last["best_params"] = theta
            shutil.rmtree(best_dir, ignore_errors=True)
            shutil.copytree(local_dir, best_dir)

        comm = ""
        if len(history["bytes_total"]) >= server_round:
            comm = (f" | comm={history['bytes_total'][server_round - 1] / MB:.3f}MB"
                    f" (cum {history['cum_mb'][server_round - 1]:.3f}MB)")
        print(f"[{tag}] round {server_round:02d} | loss={loss:.4f} "
              f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f} "
              f"ap={m['ap']:.4f}{comm}")
        if on_round is not None:
            on_round(server_round, num_rounds, loss, m)
        return loss, {k: (0.0 if np.isnan(v) else v) for k, v in m.items()}

    # parameter awal = theta saja; di-seed agar sama antar metode/run
    set_seed(seed)
    init_model = ClientModel(common_dim, common_dim, n_classes)  # n_features dummy
    init_params = ndarrays_to_parameters(get_theta_params(init_model))

    def server_fn(context: Context) -> ServerAppComponents:
        strategy = build_strategy(aggregation, num_clients, init_params,
                                  evaluate_fn, fit_metrics_agg, strat_cfg)
        return ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()      # buang sesi Ray lama agar init_args berlaku
        run_simulation(
            server_app=ServerApp(server_fn=server_fn),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=num_clients,
            backend_config={
                "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
                "init_args": {"num_cpus": max_parallel, "num_gpus": 0,
                              "include_dashboard": False},
            },
        )

        ft = dict(threshold=threshold, finetune_epochs=eval_finetune_epochs,
                  lr=lr, batch_size=batch_size, seed=seed,
                  threshold_mode=threshold_mode,
                  threshold_criterion=threshold_criterion,
                  target_sensitivity=target_sensitivity,
                  tune_split="val", collect_curves=True)

        # test: ronde TERAKHIR
        if last["params"] is not None:
            _, mean_m, per_client, ex = evaluate_personalized_per_client(
                last["params"], fed_data, common_dim, n_classes, local_dir,
                split="test", **ft)
            history["test"] = mean_m
            history["test_per_client"] = per_client
            history["test_curves"] = ex.get("curves", {})
            history["test_sweep"] = ex.get("sweep", {})
            history["test_thresholds"] = ex.get("thresholds", {})

        # test: ronde TERBAIK menurut val
        if last["best_params"] is not None and os.path.isdir(best_dir):
            _, mean_b, per_client_b, ex_b = evaluate_personalized_per_client(
                last["best_params"], fed_data, common_dim, n_classes, best_dir,
                split="test", **ft)
            history["test_best"] = mean_b
            history["test_best_per_client"] = per_client_b
            history["test_best_curves"] = ex_b.get("curves", {})
            history["test_best_sweep"] = ex_b.get("sweep", {})
            history["test_best_thresholds"] = ex_b.get("thresholds", {})
            history["best_round"] = last["best_round"]
            print(f"[{tag}] best val round (by {select_by}) = {last['best_round']}")
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)

    # ---- ringkasan bandwidth ----
    total_up = float(sum(history["bytes_up"]))
    total_down = float(sum(history["bytes_down"]))
    theta_bytes = params_nbytes(get_theta_params(init_model))
    n_rounds_done = max(len(history["bytes_total"]), 1)

    full_per_round = 0
    for c in fed_data.values():
        full_per_round += 2 * full_model_nbytes(
            ClientModel(c["n_features"], common_dim, n_classes))   # down + up
    full_total = full_per_round * n_rounds_done
    actual_total = total_up + total_down
    saving_pct = (1 - actual_total / full_total) * 100 if full_total else 0.0

    history["comm"] = {
        "theta_size_mb": theta_bytes / MB,
        "total_up_mb": total_up / MB,
        "total_down_mb": total_down / MB,
        "total_mb": actual_total / MB,
        "avg_per_round_mb": actual_total / MB / n_rounds_done,
        "avg_per_client_mb": actual_total / MB / max(num_clients, 1),
        "full_total_mb": full_total / MB,
        "saving_pct": saving_pct,
    }

    if verbose_report:
        history["summary"] = report_results(history, client_label)
    return history


# -----------------------------------------------------------------------------
# Komparasi antar agregasi (fedavg vs fedprox vs pairwise)
# -----------------------------------------------------------------------------
def run_comparison(fed_data: Dict[str, dict], name: str, common_dim: int,
                   aggregations=VALID_AGGS, seeds=(42,),
                   **run_kwargs) -> Dict[str, Dict]:
    """
    Jalankan beberapa agregasi pada data & seed yang sama.
    Hasil tiap seed di histories[agg]["runs"]; history utama = seed pertama.
    Argumen lain (mode, fedprox_mu, num_rounds, threshold_mode, ...) diteruskan
    ke run_federated.
    """
    run_kwargs.setdefault("verbose_report", False)
    histories = {}
    for agg in aggregations:
        runs = []
        for s in seeds:
            print(f"\n===== {name} | AGG: {agg} | seed={s} =====")
            runs.append(run_federated(fed_data, name, common_dim,
                                      aggregation=agg, seed=s, **run_kwargs))
        h = dict(runs[0])      # salinan dangkal: hindari referensi melingkar
        h["runs"] = runs
        histories[agg] = h
    return histories


def summarize_seeds(histories: Dict[str, Dict], split: str = "test") -> Dict:
    """mean, std antar seed untuk metrik rata-rata client. split: 'test' | 'test_best'."""
    out = {}
    for agg, h in histories.items():
        runs = [r for r in h.get("runs", [h]) if r.get(split)]
        out[agg] = {}
        for k in METRIC_KEYS:
            v = np.array([r[split][k] for r in runs], dtype=float)
            if v.size == 0 or np.all(np.isnan(v)):
                out[agg][k] = (float("nan"), float("nan"))
            else:
                out[agg][k] = (float(np.nanmean(v)), float(np.nanstd(v)))
    return out


def print_seed_summary(histories: Dict[str, Dict], split: str = "test") -> None:
    summ = summarize_seeds(histories, split)
    rows = {agg: {k: f"{mu:.4f} ± {sd:.4f}" for k, (mu, sd) in d.items()}
            for agg, d in summ.items()}
    print(f"\n=== Ringkasan antar seed ({split}) ===")
    print(pd.DataFrame(rows).T.to_string())


def build_client_comparison(histories: Dict[str, Dict],
                            split: str = "test_per_client",
                            baseline: str = "fedavg") -> list:
    methods = list(histories.keys())
    clients = []
    for h in histories.values():
        for cn in h.get(split, {}):
            if cn not in clients:
                clients.append(cn)

    rows = []
    for cn in clients + ["MEAN"]:
        for k in METRIC_KEYS:
            row = {"client": cn, "metric": k}
            for meth in methods:
                pc = histories[meth].get(split, {})
                if cn == "MEAN":
                    row[meth] = _nanmean(m[k] for m in pc.values()) if pc else float("nan")
                else:
                    row[meth] = pc.get(cn, {}).get(k, float("nan"))
            if baseline in methods:
                for meth in methods:
                    if meth != baseline:
                        row[f"delta_{meth}_vs_{baseline}"] = row[meth] - row[baseline]
            rows.append(row)
    return rows


def print_comparison(histories: Dict[str, Dict],
                     split: str = "test_per_client",
                     baseline: str = "fedavg") -> None:
    rows = build_client_comparison(histories, split, baseline)
    if not rows:
        print("[compare] tidak ada data untuk dibandingkan.")
        return
    methods = list(histories.keys())
    deltas = [m for m in methods if m != baseline and baseline in methods]

    header = f"{'client':<22}{'metric':<11}" + "".join(f"{m:>13}" for m in methods) \
        + "".join(f"{'Δ ' + d:>16}" for d in deltas)
    print("\n" + header)
    print("-" * len(header))
    prev = None
    for r in rows:
        if prev is not None and r["client"] != prev:
            print()
        prev = r["client"]
        line = f"{r['client'][:21]:<22}{r['metric']:<11}"
        line += "".join(f"{r[m]:>13.4f}" for m in methods)
        line += "".join(f"{r[f'delta_{d}_vs_{baseline}']:>+16.4f}" for d in deltas)
        print(line)


def export_comparison_csv(histories: Dict[str, Dict], out_path: str,
                          split: str = "test_per_client",
                          baseline: str = "fedavg") -> str:
    rows = build_client_comparison(histories, split, baseline)
    if not rows:
        return ""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[export] komparasi per-client tersimpan di: {out_path}")
    return out_path


def export_rounds_comparison_csv(histories: Dict[str, Dict], out_path: str) -> str:
    """Kurva per-ronde (mean antar client) semua agregasi dalam satu CSV (long)."""
    fieldnames = ["aggregation", "round", "loss", *METRIC_KEYS,
                  "n_client_fit", "cum_MB"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for agg, h in histories.items():
            for i, r in enumerate(h.get("round", [])):
                row = {"aggregation": agg, "round": r, "loss": h["loss"][i]}
                for k in METRIC_KEYS:
                    row[k] = h[k][i]
                row["n_client_fit"] = (h["n_client_fit"][i]
                                       if i < len(h["n_client_fit"]) else "")
                row["cum_MB"] = h["cum_mb"][i] if i < len(h["cum_mb"]) else ""
                writer.writerow(row)
    print(f"[export] kurva per-ronde semua agregasi tersimpan di: {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Ekspor CSV (single run)
# -----------------------------------------------------------------------------
def export_history_csv(history: Dict, out_path: str) -> str:
    n = len(history.get("round", []))
    fieldnames = ["round", "loss", *METRIC_KEYS, "n_client_fit",
                  "bytes_up_MB", "bytes_down_MB", "bytes_total_MB", "cum_MB"]

    def mb(key, i):
        return history[key][i] / MB if i < len(history[key]) else ""

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            row = {"round": history["round"][i], "loss": history["loss"][i]}
            for k in METRIC_KEYS:
                row[k] = history[k][i]
            row["n_client_fit"] = (history["n_client_fit"][i]
                                   if i < len(history["n_client_fit"]) else "")
            row["bytes_up_MB"] = mb("bytes_up", i)
            row["bytes_down_MB"] = mb("bytes_down", i)
            row["bytes_total_MB"] = mb("bytes_total", i)
            row["cum_MB"] = history["cum_mb"][i] if i < len(history["cum_mb"]) else ""
            writer.writerow(row)
    print(f"[export] metrik per-ronde tersimpan di: {out_path}")
    return out_path


def export_per_client_csv(history: Dict, out_path: str,
                          split: str = "val_per_client") -> str:
    """split: 'val_per_client' (per ronde) | 'test_per_client' | 'test_best_per_client'."""
    if split in ("test_per_client", "test_best_per_client"):
        pc = history.get(split)
        if not pc:
            return ""
        per_client_table(pc).to_csv(out_path)
        print(f"[export] metrik {split} tersimpan di: {out_path}")
        return out_path

    rows = []
    for r_idx, per_client in zip(history["round"], history.get(split, [])):
        for cname, m in per_client.items():
            rows.append({
                "round": r_idx, "client": cname, "n": m.get("n", ""),
                **{k: m[k] for k in METRIC_KEYS},
            })
    if not rows:
        return ""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "client", "n", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[export] metrik per-client tersimpan di: {out_path}")
    return out_path


def export_curves_csv(history: Dict, out_path: str,
                      split: str = "test_best_curves") -> str:
    """Titik kurva ROC & PR (long): client, curve, x, y, auc/ap, prevalence."""
    curves = history.get(split)
    if not curves:
        return ""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client", "curve", "x", "y", "auc", "ap", "prevalence", "n"])
        for cn, cv in curves.items():
            for x, y in zip(cv["fpr"], cv["tpr"]):
                w.writerow([cn, "roc(fpr,tpr)", x, y, cv["auc"], cv["ap"],
                            cv["prevalence"], cv["n"]])
            for x, y in zip(cv["recall"], cv["precision"]):
                w.writerow([cn, "pr(recall,precision)", x, y, cv["auc"], cv["ap"],
                            cv["prevalence"], cv["n"]])
    print(f"[export] titik kurva ROC/PR tersimpan di: {out_path}")
    return out_path


def export_threshold_sweep_csv(history: Dict, out_path: str,
                               split: str = "test_best_sweep") -> str:
    """Tabel threshold sweep semua client (+ POOLED) dalam satu CSV."""
    sweeps = history.get(split)
    if not sweeps:
        return ""
    fields = ["client", "threshold", "precision", "recall", "specificity",
              "f1", "youden", "pos_pred_rate"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cn, rows in sweeps.items():
            for r in rows:
                writer.writerow({"client": cn, **r})
    print(f"[export] threshold sweep tersimpan di: {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Contoh pemakaian
# -----------------------------------------------------------------------------
# histories = run_comparison(
#     fed_data, "exp1", common_dim=64,
#     aggregations=("fedavg", "fedprox", "pairwise"),
#     seeds=(0, 1, 2),
#     mode="sim",              # atau "alt"
#     fedprox_mu=0.01,         # mu khusus FedProx
#     num_rounds=10, local_epochs=5,
#     select_by="ap",          # pilih ronde terbaik via AP (cocok data imbalance)
#     threshold_mode="per_client",   # threshold dipilih dari VAL per client
#     threshold_criterion="f1",      # "f1" | "youden" | "gmean" | "sens_target"
#     # target_sensitivity=0.9,      # bila criterion="sens_target"
# )
# print_comparison(histories, split="test_best_per_client")  # ronde terbaik (val)
# print_comparison(histories, split="test_per_client")       # ronde terakhir
# print_seed_summary(histories, split="test_best")           # mean ± std antar seed
# export_comparison_csv(histories, "compare_test.csv", split="test_best_per_client")
# export_rounds_comparison_csv(histories, "compare_rounds.csv")
#
# # kurva diagnostik (seed pertama):
# h = histories["pairwise"]
# plot_diagnostic_curves(h, "roc_pr_pairwise.png", split="test_best_curves")
# plot_comparison_curves(histories, "roc_pr_compare.png")
# print_threshold_sweep(h, client="__POOLED__", split="test_best_sweep")
# export_curves_csv(h, "curves_pairwise.csv")
# export_threshold_sweep_csv(h, "sweep_pairwise.csv")
#
# # single run:
# h = run_federated(fed_data, "exp1", 64, aggregation="fedprox", fedprox_mu=0.05)