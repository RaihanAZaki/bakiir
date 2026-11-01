
from pathlib import Path
from collections import defaultdict
import hashlib
import random
import shutil

import pandas as pd

SEED = 42
random.seed(SEED)

REPORTS_DIR = Path("reports")
OUTPUT_DIR = Path("dataset/cleaned")

inventory = pd.read_csv(
    REPORTS_DIR / "dataset_inventory.csv"
)

def calculate_hash(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()

# Kelompokkan file berdasarkan isi identik
hash_groups = defaultdict(list)

for row in inventory.to_dict("records"):
    path = Path(row["path"])

    if not path.is_file():
        raise FileNotFoundError(path)

    file_hash = calculate_hash(path)
    hash_groups[file_hash].append(row)

# Jangan lanjut jika gambar identik
# mempunyai label kesegaran berbeda
for file_hash, rows in hash_groups.items():
    labels = {r["freshness"] for r in rows}

    if len(labels) > 1:
        raise ValueError(
            f"Konflik label pada hash {file_hash}"
        )

# Kelompokkan berdasarkan kelas dan jenis ayam
strata = defaultdict(list)

for file_hash, rows in hash_groups.items():
    species = {r["chicken_type"] for r in rows}

    if len(species) > 1:
        raise ValueError(
            f"Konflik jenis ayam pada hash {file_hash}"
        )

    representative = rows[0]
    key = (
        representative["freshness"],
        representative["chicken_type"]
    )

    strata[key].append((file_hash, representative))

# Hindari hasil lama tercampur
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} sudah ada. "
        "Pindahkan atau hapus folder hasil sebelumnya "
        "setelah memastikan tidak diperlukan."
    )

OUTPUT_DIR.mkdir(parents=True)

manifest = []

for (freshness, species), items in sorted(strata.items()):
    items.sort(key=lambda item: item[0])
    random.shuffle(items)

    total = len(items)
    train_end = int(total * 0.70)
    valid_end = train_end + int(total * 0.15)

    groups = {
        "train": items[:train_end],
        "valid": items[train_end:valid_end],
        "test": items[valid_end:]
    }

    for split, selected_items in groups.items():
        destination = OUTPUT_DIR / split / freshness
        destination.mkdir(parents=True, exist_ok=True)

        for file_hash, row in selected_items:
            source = Path(row["path"])

            filename = (
                f"{species}_{file_hash[:16]}"
                f"{source.suffix.lower()}"
            )

            target = destination / filename
            shutil.copy2(source, target)

            manifest.append({
                "split": split,
                "freshness": freshness,
                "chicken_type": species,
                "sha256": file_hash,
                "source_path": str(source),
                "output_path": str(target)
            })

result = pd.DataFrame(manifest)

# Validasi tidak ada hash lintas split
cross_split = (
    result.groupby("sha256")["split"]
    .nunique()
)

assert cross_split.max() == 1

REPORTS_DIR.mkdir(exist_ok=True)

result.to_csv(
    REPORTS_DIR / "cleaned_dataset_manifest.csv",
    index=False
)

print("\n=== HASIL CLEANING ===")
print(f"Gambar asli: {len(inventory)}")
print(f"Gambar unik: {len(result)}")
print(f"Duplikasi dibuang: {len(inventory) - len(result)}")

print("\n=== DISTRIBUSI DATA ===")
print(
    pd.crosstab(
        result["split"],
        result["freshness"]
    ).to_string()
)

print("\nDuplikasi hash lintas split: 0")
print(f"Dataset baru: {OUTPUT_DIR}")
print("\nCleaning selesai.")
