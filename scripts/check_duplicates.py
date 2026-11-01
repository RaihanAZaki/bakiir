
from pathlib import Path
from collections import defaultdict
import hashlib
import pandas as pd

DATASET_DIR = Path("dataset/raw/Egg Freshness")
REPORTS_DIR = Path("reports")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

inventory = pd.read_csv(
    REPORTS_DIR / "dataset_inventory.csv"
)

hashes = defaultdict(list)
missing_files = []

def get_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

print("Memulai pemeriksaan duplikasi...")

for _, row in inventory.iterrows():
    path = Path(row["path"])

    if not path.exists():
        missing_files.append(str(path))
        continue

    file_hash = get_sha256(path)

    hashes[file_hash].append({
        "split": row["split"],
        "freshness": row["freshness"],
        "chicken_type": row["chicken_type"],
        "filename": row["filename"],
        "path": str(path),
    })

duplicates = []
cross_split_duplicates = []
conflicting_labels = []

for file_hash, items in hashes.items():
    if len(items) < 2:
        continue

    for item in items:
        duplicates.append({
            "hash": file_hash,
            **item
        })

    splits = {item["split"] for item in items}
    labels = {item["freshness"] for item in items}

    if len(splits) > 1:
        for item in items:
            cross_split_duplicates.append({
                "hash": file_hash,
                **item
            })

    if len(labels) > 1:
        for item in items:
            conflicting_labels.append({
                "hash": file_hash,
                **item
            })

def save_report(rows, filename):
    pd.DataFrame(rows).to_csv(
        REPORTS_DIR / filename,
        index=False
    )

save_report(duplicates, "duplicate_images.csv")
save_report(
    cross_split_duplicates,
    "cross_split_duplicates.csv"
)
save_report(
    conflicting_labels,
    "conflicting_labels.csv"
)

print("\n=== HASIL PEMERIKSAAN ===")
print(f"Total gambar: {len(inventory)}")
print(f"File berhasil diperiksa: {sum(map(len, hashes.values()))}")
print(f"File tidak ditemukan: {len(missing_files)}")

print(
    "Grup gambar identik:",
    sum(len(items) > 1 for items in hashes.values())
)

print(
    "Grup identik lintas split:",
    sum(
        len({item["split"] for item in items}) > 1
        for items in hashes.values()
    )
)

print(
    "Grup dengan label berbeda:",
    sum(
        len({item["freshness"] for item in items}) > 1
        for items in hashes.values()
    )
)

if missing_files:
    print("\nFile tidak ditemukan (5 pertama):")
    for path in missing_files[:5]:
        print(path)

print("\nLaporan tersimpan di folder reports/")
    