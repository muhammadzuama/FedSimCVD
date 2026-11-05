"""Sisi CLIENT Flower (FedAvg / FedProx; FedPairwise & FedSim = sisi server saja)."""
import torch
import torch.nn as nn
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader, TensorDataset

import config
from data_utils import load_all, state_list
from evaluation import evaluate_model
from model import build_model
from utils import (class_weights, get_seed, get_weights, log_error,
                   set_weights, trainable_params)


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
