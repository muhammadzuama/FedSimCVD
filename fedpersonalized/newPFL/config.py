# config.py
from pathlib import Path
import torch

# Data
CLIENT_DIR = Path(r'C:\Users\s2\Documents\FIXRKI\bfprssprep\split\clients')

# Output (absolut agar server & client Ray menulis ke folder yang sama)
OUT_DIR = Path('outputs').resolve()


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# Perangkat & reproduksibilitas
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# Training lokal
LR = 1e-3
BATCH_SIZE = 64
LOCAL_EPOCHS = 10

# Federated
NUM_ROUNDS = 10
FRACTION_FIT = 1.0

# Metrik
METRIK = ['acc', 'prec', 'rec', 'f1', 'auc']
NAMA_METRIK = {
    'acc': 'Akurasi',
    'prec': 'Presisi',
    'rec': 'Recall',
    'f1': 'F1',
    'auc': 'AUC',
}