"""
model.py

Definisi model (theta dan phi) untuk klasifikasi Drebin menggunakan LSTM.
- theta -> FeatureExtractor (LSTM)
- phi   -> ClassifierHead

Input tetap berbentuk (batch, in_dim), reshape ke sekuens dilakukan
di dalam FeatureExtractor sehingga pipeline data tidak perlu diubah.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Bagian theta: mengekstrak representasi fitur dengan LSTM."""

    def __init__(self, in_dim, hidden_dim=64, out_dim=32,
                 num_layers=1, seq_len=16, mode="chunk", dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.mode = mode

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

        self.lstm = nn.LSTM(
            input_size=self.chunk_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, in_dim)
        pad = self.padded_dim - x.size(1)
        if pad > 0:
            x = F.pad(x, (0, pad))
        x = x.view(x.size(0), self.seq_len, self.chunk_size)

        _, (h_n, _) = self.lstm(x)      # h_n: (num_layers, batch, hidden_dim)
        h_last = h_n[-1]                # hidden state layer terakhir
        return self.relu(self.fc(h_last))


class ClassifierHead(nn.Module):
    """Bagian phi: mengklasifikasikan fitur menjadi kelas output."""

    def __init__(self, in_dim=32, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class LSTMModel(nn.Module):
    """Model gabungan: theta (LSTM FeatureExtractor) + phi (ClassifierHead)."""

    def __init__(self, in_dim, num_classes=2, hidden_dim=64, out_dim=32,
                 num_layers=1, seq_len=16, mode="chunk", dropout=0.0):
        super().__init__()
        self.theta = FeatureExtractor(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=num_layers, seq_len=seq_len, mode=mode, dropout=dropout,
        )
        self.phi = ClassifierHead(in_dim=out_dim, num_classes=num_classes)

    def forward(self, x):
        features = self.theta(x)
        logits = self.phi(features)
        return logits


# Alias agar kode lama yang memanggil FFNNModel tetap jalan
FFNNModel = LSTMModel


def build_model(in_dim, num_classes=2, seed=42, **kwargs):
    torch.manual_seed(seed)
    model = LSTMModel(in_dim=in_dim, num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":
    example_in_dim = 100
    model = build_model(in_dim=example_in_dim)
    print(model)

    x = torch.rand(8, example_in_dim)
    print(model(x).shape)  # (8, 2)