"""
LabelReverse.py
---------------
Membalik label biner pada file train.csv dan val.csv: 0 -> 1 dan 1 -> 0.

Nama folder client mengikuti SUFFIX:
    SUFFIX = ""          -> state_53
    SUFFIX = "sinatetic" -> state_53_sinatetic

Cara pakai dari file lain:

    from LabelReverse import reverse_client_labels, compare_outputs

    reverse_client_labels(
        clients_dir=r"C:\\...\\ScenarioMalacious\\ML2\\clientsMalacious_Selected",
        label_col="_michd",
        output_dir=r"C:\\...\\ScenarioMalacious\\ML2\\clientsMalacious_Reverse",
    )

Atau untuk satu file saja:

    from LabelReverse import reverse_csv_labels
    reverse_csv_labels(r"C:\\...\\state_04\\train.csv", label_col="_michd")
"""

from pathlib import Path
import re
import shutil

import pandas as pd

DEFAULT_SPLITS = ("train", "val")
DEFAULT_LABEL_COL = "label"
MARKER_NAME = ".labels_reversed"  # penanda agar tidak ter-reverse dua kali
SUFFIX = ""                       # "" -> state_53 | "sinatetic" -> state_53_sinatetic


def state_folder(s: int, suffix: str = None) -> str:
    """Nama folder untuk nomor state, contoh: 4 -> state_04 atau state_04_sinatetic."""
    suffix = SUFFIX if suffix is None else suffix
    return f"state_{s:02d}" + (f"_{suffix}" if suffix else "")


def flip_labels(df: pd.DataFrame, label_col: str = DEFAULT_LABEL_COL) -> pd.DataFrame:
    """Kembalikan salinan DataFrame dengan label 0 <-> 1."""
    if label_col not in df.columns:
        raise KeyError(
            f"Kolom '{label_col}' tidak ada. Kolom tersedia: {list(df.columns)}"
        )

    values = set(df[label_col].dropna().unique())
    if not values.issubset({0, 1}):
        raise ValueError(
            f"Kolom '{label_col}' berisi nilai selain 0/1: {sorted(values)}"
        )

    out = df.copy()
    out[label_col] = 1 - out[label_col].astype(int)
    return out


def reverse_csv_labels(
    csv_path,
    label_col: str = DEFAULT_LABEL_COL,
    output_path=None,
    backup: bool = True,
) -> Path:
    """
    Balik label pada satu file CSV.

    output_path=None -> timpa file asli (dengan backup *.bak jika backup=True).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"File tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)
    before = df[label_col].value_counts().sort_index().to_dict() if label_col in df else {}
    flipped = flip_labels(df, label_col)
    after = flipped[label_col].value_counts().sort_index().to_dict()

    if output_path is None:
        target = csv_path
        if backup:
            bak = csv_path.with_suffix(csv_path.suffix + ".bak")
            if not bak.exists():  # jangan timpa backup asli
                shutil.copy2(csv_path, bak)
    else:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

    flipped.to_csv(target, index=False)
    print(f"  {csv_path.name}: {before} -> {after}")
    return target


def find_states(clients_dir, suffix: str = None) -> list:
    """Cari semua folder state_XX (atau state_XX_<suffix>) di dalam clients_dir."""
    suffix = SUFFIX if suffix is None else suffix
    pattern = rf"state_(\d+)_{re.escape(suffix)}" if suffix else r"state_(\d+)"
    clients_dir = Path(clients_dir)
    return sorted(
        int(m.group(1))
        for p in clients_dir.iterdir()
        if p.is_dir() and (m := re.fullmatch(pattern, p.name))
    )


def reverse_client_labels(
    clients_dir,
    label_col: str = DEFAULT_LABEL_COL,
    states=None,
    splits=DEFAULT_SPLITS,
    output_dir=None,
    backup: bool = True,
    force: bool = False,
) -> None:
    """
    Balik label train.csv & val.csv untuk semua (atau sebagian) client.

    clients_dir : folder berisi state_XX/ (atau state_XX_<suffix>/)
    states      : list nomor state; None = semua state yang ada
    splits      : ("train", "val") -> mencari train.csv dan val.csv
    output_dir  : None = timpa file asli; isi path = simpan hasil ke folder baru
    backup      : buat train.csv.bak / val.csv.bak saat menimpa file asli
    force       : True = abaikan penanda 'sudah di-reverse'
    """
    clients_dir = Path(clients_dir)
    if not clients_dir.is_dir():
        raise FileNotFoundError(f"Folder tidak ditemukan: {clients_dir}")

    states = find_states(clients_dir) if states is None else list(states)
    if not states:
        print(f"⚠️ Tidak ada folder state_XX{'_' + SUFFIX if SUFFIX else ''} di {clients_dir}")
        return

    in_place = output_dir is None
    done = skipped = 0

    for s in states:
        src_dir = clients_dir / state_folder(s)
        if not src_dir.is_dir():
            print(f"⚠️ Folder tidak ada: {src_dir}")
            continue

        marker = src_dir / MARKER_NAME
        if in_place and marker.exists() and not force:
            print(f"⏭️  {state_folder(s)} sudah pernah di-reverse, dilewati (pakai force=True untuk memaksa)")
            skipped += 1
            continue

        print(state_folder(s))
        dst_dir = src_dir if in_place else Path(output_dir) / state_folder(s)

        for split in splits:
            reverse_csv_labels(
                src_dir / f"{split}.csv",
                label_col=label_col,
                output_path=None if in_place else dst_dir / f"{split}.csv",
                backup=backup,
            )

        if in_place:
            marker.write_text("labels reversed (0<->1)\n")
        done += 1

    print(f"\nSelesai. Client diproses: {done} | dilewati: {skipped}")


def _count_labels(path, label_col):
    """Hitung jumlah label 0 dan 1 pada satu CSV."""
    df = pd.read_csv(path, usecols=[label_col])
    vc = df[label_col].value_counts()
    return int(vc.get(0, 0)), int(vc.get(1, 0))


def show_label_summary(
    clients_dir,
    label_col: str = DEFAULT_LABEL_COL,
    states=None,
    splits=DEFAULT_SPLITS,
) -> pd.DataFrame:
    """
    Tampilkan jumlah label 0 dan 1 per state & split (kondisi saat ini).
    Kalau file .bak ada (hasil reverse in-place), ditampilkan juga kondisi
    SEBELUM sehingga perubahannya terlihat berdampingan.
    """
    clients_dir = Path(clients_dir)
    states = find_states(clients_dir) if states is None else list(states)

    rows = []
    for s in states:
        for split in splits:
            path = clients_dir / state_folder(s) / f"{split}.csv"
            if not path.is_file():
                continue
            bak = path.with_suffix(path.suffix + ".bak")

            now0, now1 = _count_labels(path, label_col)
            row = {"state": f"{s:02d}", "split": split}
            if bak.is_file():
                b0, b1 = _count_labels(bak, label_col)
                row.update({"sebelum_0": b0, "sebelum_1": b1})
            row.update({"sesudah_0": now0, "sesudah_1": now1})
            if bak.is_file():
                row["terbalik_benar"] = (b0 == now1) and (b1 == now0)
            rows.append(row)

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    return summary


def compare_outputs(
    original_dir,
    reversed_dir,
    label_col: str = DEFAULT_LABEL_COL,
    states=None,
    splits=DEFAULT_SPLITS,
) -> pd.DataFrame:
    """
    Bandingkan folder asli vs folder hasil reverse (kalau pakai output_dir).
    Cek juga bahwa setiap baris benar-benar terbalik labelnya.
    """
    original_dir, reversed_dir = Path(original_dir), Path(reversed_dir)
    if not reversed_dir.is_dir():
        raise FileNotFoundError(
            f"Folder hasil reverse belum ada: {reversed_dir}. "
            "Jalankan reverse_client_labels(..., output_dir=...) terlebih dahulu."
        )
    states = find_states(reversed_dir) if states is None else list(states)

    rows = []
    for s in states:
        for split in splits:
            a = original_dir / state_folder(s) / f"{split}.csv"
            b = reversed_dir / state_folder(s) / f"{split}.csv"
            if not (a.is_file() and b.is_file()):
                continue
            da, db = pd.read_csv(a), pd.read_csv(b)
            a0, a1 = int((da[label_col] == 0).sum()), int((da[label_col] == 1).sum())
            b0, b1 = int((db[label_col] == 0).sum()), int((db[label_col] == 1).sum())
            all_flipped = len(da) == len(db) and bool(
                ((da[label_col] + db[label_col]) == 1).all()
            )
            rows.append({
                "state": f"{s:02d}", "split": split,
                "asli_0": a0, "asli_1": a1,
                "baru_0": b0, "baru_1": b1,
                "semua_baris_terbalik": all_flipped,
            })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    # Contoh pemakaian langsung: sesuaikan path dan nama kolom label.
    BASE = Path(r"C:\Users\s2\Documents\FIXRKI\FIXbanget\ScenarioMalacious\ML2")
    CLIENTS_DIR = BASE / "clientsMalacious_Selected"
    NEW_DIR = BASE / "clientsMalacious_Reverse"

    reverse_client_labels(
        clients_dir=CLIENTS_DIR,
        label_col="_michd",
        output_dir=NEW_DIR,   # hapus baris ini jika ingin menimpa file asli
    )
    compare_outputs(CLIENTS_DIR, NEW_DIR, label_col="_michd")