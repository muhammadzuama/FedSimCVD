"""
model.py

Definisi model (theta dan phi) untuk klasifikasi Drebin.
- theta -> FeatureExtractor
- phi   -> ClassifierHead

Nama layer dibuat eksplisit (fc1, fc2, fc3) agar mudah diakses untuk analisis bobot.
"""

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    Bagian theta: mengekstrak representasi fitur dari input.

    Dasar desain:
    - Fitur Drebin berupa vektor biner sparse berdimensi tinggi (ribuan fitur),
      sehingga dibutuhkan kompresi bertahap (funnel): in_dim -> 512 -> 256 -> 128.
      Penyempitan bertahap memaksa jaringan mempelajari kombinasi fitur
      (mis. permission + API call yang muncul bersamaan) alih-alih menghafal.
    - BatchNorm1d menstabilkan distribusi aktivasi tiap layer sehingga training
      lebih cepat dan tidak sensitif terhadap inisialisasi.
    - Dropout mengurangi overfitting, yang rawan terjadi pada data sparse
      berdimensi tinggi dengan sampel terbatas.
    - Urutan Linear -> BatchNorm -> ReLU -> Dropout adalah pola standar.
    """

    def __init__(self, in_dim, hidden_dim=512, mid_dim=256, out_dim=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, mid_dim)
        self.bn2 = nn.BatchNorm1d(mid_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(mid_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        return x


class ClassifierHead(nn.Module):
    """
    Bagian phi: mengklasifikasikan fitur menjadi kelas output.

    Dasar desain:
    - Head sebelumnya hanya satu layer linear (batas keputusan linear di ruang
      fitur). Menambah satu hidden layer memungkinkan head mempelajari batas
      keputusan non-linear, berguna ketika kelas malware/benign tidak
      terpisah linear di ruang representasi theta.
    - Dimensi menyempit bertahap (128 -> 64 -> num_classes) agar representasi
      dipadatkan sebelum keputusan akhir.
    - Dropout pada head mengurangi ketergantungan pada neuron tertentu.
    - Output berupa logits (tanpa softmax) karena CrossEntropyLoss sudah
      menerapkan log-softmax secara internal.
    """

    def __init__(self, in_dim=128, hidden_dim=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        x = self.drop1(self.relu1(self.bn1(self.fc1(features))))
        return self.fc(x)


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