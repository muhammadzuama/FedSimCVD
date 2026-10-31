"""
model.py

Definisi model (theta dan phi) untuk klasifikasi Drebin.
- theta -> FeatureExtractor
- phi   -> ClassifierHead

Nama layer dibuat eksplisit (fc1, fc2) agar mudah diakses untuk analisis bobot.
"""

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Bagian theta: mengekstrak representasi fitur dari input."""

    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x


class ClassifierHead(nn.Module):
    """Bagian phi: mengklasifikasikan fitur menjadi kelas output."""

    def __init__(self, in_dim=32, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class FFNNModel(nn.Module):
    """Model gabungan: theta (FeatureExtractor) + phi (ClassifierHead)."""

    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.theta = FeatureExtractor(in_dim=in_dim)
        self.phi = ClassifierHead(num_classes=num_classes)

    def forward(self, x):
        features = self.theta(x)
        logits = self.phi(features)
        return logits


def build_model(in_dim, num_classes=2, seed=42):
    """
    Helper untuk membangun model dengan seed tetap agar reproducible.

    Args:
        in_dim (int): dimensi fitur input (mis. X_train_t.shape[1]).
        num_classes (int): jumlah kelas output.
        seed (int): random seed untuk inisialisasi bobot.

    Returns:
        FFNNModel
    """
    torch.manual_seed(seed)
    model = FFNNModel(in_dim=in_dim, num_classes=num_classes)
    return model


if __name__ == "__main__":
    # Contoh pemakaian mandiri (butuh in_dim manual jika dijalankan langsung)
    example_in_dim = 100  # ganti sesuai jumlah fitur dataset Anda
    model = build_model(in_dim=example_in_dim)
    print(model)

