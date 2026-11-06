"""
model.py

Definisi model (theta dan phi) untuk klasifikasi Drebin menggunakan RNN
(LSTM atau GRU).
- theta -> FeatureExtractor (LSTM/GRU)
- phi   -> ClassifierHead

Input tetap berbentuk (batch, in_dim), reshape ke sekuens dilakukan
di dalam FeatureExtractor sehingga pipeline data tidak perlu diubah.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Bagian theta: mengekstrak representasi fitur dengan LSTM atau GRU."""

    def __init__(self, in_dim, hidden_dim=64, out_dim=32,
                 num_layers=1, seq_len=16, mode="chunk",
                 rnn_type="lstm", dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.mode = mode
        self.rnn_type = rnn_type.lower()

        if mode == "feature_seq":
            # Tiap fitur = 1 timestep
            self.seq_len = in_dim
            self.chunk_size = 1
        elif mode == "chunk":
            # Vektor dipotong menjadi seq_len chunk (dengan padding nol bila perlu)
            self.seq_len = seq_len
            self.chunk_size = math.ceil(in_dim / seq_len)
        else:
            raise ValueError("mode harus 'chunk' atau 'feature_seq'")

        self.padded_dim = self.seq_len * self.chunk_size

        rnn_kwargs = dict(
            input_size=self.chunk_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            raise ValueError("rnn_type harus 'lstm' atau 'gru'")

        self.fc = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, in_dim)
        pad = self.padded_dim - x.size(1)
        if pad > 0:
            x = F.pad(x, (0, pad))
        x = x.view(x.size(0), self.seq_len, self.chunk_size)

        _, h = self.rnn(x)
        # LSTM mengembalikan (h_n, c_n), GRU hanya h_n
        h_n = h[0] if self.rnn_type == "lstm" else h  # (num_layers, batch, hidden_dim)
        h_last = h_n[-1]                              # hidden state layer terakhir
        return self.relu(self.fc(h_last))


class ClassifierHead(nn.Module):
    """Bagian phi: mengklasifikasikan fitur menjadi kelas output."""

    def __init__(self, in_dim=32, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class RNNModel(nn.Module):
    """Model gabungan: theta (LSTM/GRU FeatureExtractor) + phi (ClassifierHead)."""

    def __init__(self, in_dim, num_classes=2, hidden_dim=64, out_dim=32,
                 num_layers=1, seq_len=16, mode="chunk",
                 rnn_type="lstm", dropout=0.0):
        super().__init__()
        self.theta = FeatureExtractor(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=num_layers, seq_len=seq_len, mode=mode,
            rnn_type=rnn_type, dropout=dropout,
        )
        self.phi = ClassifierHead(in_dim=out_dim, num_classes=num_classes)

    def forward(self, x):
        features = self.theta(x)
        logits = self.phi(features)
        return logits


# Alias agar kode lama tetap jalan
LSTMModel = RNNModel
FFNNModel = RNNModel


def build_model(in_dim, num_classes=2, seed=42, **kwargs):
    torch.manual_seed(seed)
    model = RNNModel(in_dim=in_dim, num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":
    example_in_dim = 100
    x = torch.rand(8, example_in_dim)

    for rnn_type in ["lstm", "gru"]:
        model = build_model(in_dim=example_in_dim, rnn_type=rnn_type)
        print(f"--- {rnn_type.upper()} ---")
        print(model)
        print(model(x).shape)  # (8, 2)