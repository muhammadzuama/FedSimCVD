"""
model.py

Definisi model (theta dan phi) untuk klasifikasi Drebin dengan arsitektur CNN-LSTM.
- theta -> FeatureExtractor (Conv1d x2 -> LSTM)
- phi   -> ClassifierHead

Input Drebin berupa vektor fitur biner (batch, in_dim). Vektor ini diperlakukan
sebagai "sinyal 1D" dengan 1 channel: (batch, 1, in_dim). CNN mengekstrak pola
lokal, lalu LSTM membaca hasilnya sebagai urutan.

Nama layer eksplisit (conv1, conv2, lstm, fc) agar mudah diakses untuk analisis bobot.
"""

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Bagian theta: CNN 1D + LSTM untuk mengekstrak representasi fitur."""

    def __init__(self, in_dim, conv_channels=(16, 32), kernel_size=5,
                 pool_size=2, lstm_hidden=32, out_dim=32):
        super().__init__()
        c1, c2 = conv_channels
        pad = kernel_size // 2

        # Blok CNN
        self.conv1 = nn.Conv1d(1, c1, kernel_size, padding=pad)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(pool_size)

        self.conv2 = nn.Conv1d(c1, c2, kernel_size, padding=pad)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(pool_size)

        # Blok LSTM: membaca urutan hasil CNN
        self.lstm = nn.LSTM(input_size=c2, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)

        # Proyeksi ke dimensi fitur akhir (sama dengan versi FFNN: 32)
        self.fc_out = nn.Linear(lstm_hidden, out_dim)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        # x: (B, in_dim) -> (B, 1, in_dim)
        x = x.unsqueeze(1)

        x = self.pool1(self.relu1(self.conv1(x)))   # (B, c1, L/p)
        x = self.pool2(self.relu2(self.conv2(x)))   # (B, c2, L/p^2)

        # Conv output (B, C, L) -> LSTM input (B, L, C)
        x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)                  # h_n: (1, B, H)
        x = h_n[-1]                                 # hidden state terakhir (B, H)

        x = self.relu3(self.fc_out(x))              # (B, out_dim)
        return x


class ClassifierHead(nn.Module):
    """Bagian phi: mengklasifikasikan fitur menjadi kelas output."""

    def __init__(self, in_dim=32, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class CNNLSTMModel(nn.Module):
    """Model gabungan: theta (CNN-LSTM FeatureExtractor) + phi (ClassifierHead)."""

    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.theta = FeatureExtractor(in_dim=in_dim)
        self.phi = ClassifierHead(num_classes=num_classes)

    def forward(self, x):
        features = self.theta(x)
        logits = self.phi(features)
        return logits


# Alias agar kode lama yang memakai nama FFNNModel tetap berjalan
FFNNModel = CNNLSTMModel


def build_model(in_dim, num_classes=2, seed=42):
    torch.manual_seed(seed)
    model = CNNLSTMModel(in_dim=in_dim, num_classes=num_classes)
    return model


if __name__ == "__main__":
    example_in_dim = 100  # ganti sesuai jumlah fitur dataset Anda
    model = build_model(in_dim=example_in_dim)
    print(model)

    # Cek bentuk output
    dummy = torch.randn(8, example_in_dim)
    print("Output shape:", model(dummy).shape)  # (8, 2)