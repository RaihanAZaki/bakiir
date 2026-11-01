
from pathlib import Path
from collections import defaultdict
from PIL import Image
import pandas as pd

# Ubah sesuai nama folder dataset kamu
DATASET_DIR = Path("dataset/raw/Egg Freshness")

SPLITS = ["train", "valid", "test"]

CHICKEN_TYPES = [
    "Chicken_Country",
    "Chicken_Red_Layer",
    "Chicken_White_Layer",
]

FRESHNESS_CLASSES = [
    "Fresh",
    "Half-Fresh",
    "Rotten",
]

records = []
invalid_images = []

for split in SPLITS:
    split_dir = DATASET_DIR / split

    if not split_dir.exists():
        print(f"WARNING: Folder tidak ditemukan: {split_dir}")
        continue

    for folder in split_dir.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name

        chicken_type = next(
            (t for t in CHICKEN_TYPES if folder_name.endswith(t)),
            None,
        )

        if chicken_type is None:
            continue

        freshness = folder_name[: -(len(chicken_type))].rstrip("_-")

        if freshness not in FRESHNESS_CLASSES:
            print(f"WARNING: Kelas tidak dikenali: {folder_name}")
            continue

        for image_path in folder.rglob("*"):
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                with Image.open(image_path) as img:
                    img.verify()

                with Image.open(image_path) as img:
                    width, height = img.size

                records.append({
                    "split": split,
                    "freshness": freshness,
                    "chicken_type": chicken_type,
                    "filename": image_path.name,
                    "path": str(image_path),
                    "width": width,
                    "height": height,
                })

            except Exception as e:
                invalid_images.append({
                    "path": str(image_path),
                    "error": str(e),
                })

df = pd.DataFrame(records)

if df.empty:
    raise SystemExit(
        "Tidak ada gambar ditemukan. Periksa DATASET_DIR dan nama folder."
    )

output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

df.to_csv(output_dir / "dataset_inventory.csv", index=False)

summary = (
    df.groupby(["split", "freshness", "chicken_type"])
    .size()
    .reset_index(name="total_images")
)

summary.to_csv(output_dir / "dataset_summary.csv", index=False)

print("\n=== JUMLAH GAMBAR PER KELAS ===")
print(df.groupby(["split", "freshness"]).size().unstack(fill_value=0))

print("\n=== JUMLAH GAMBAR PER JENIS AYAM ===")
print(summary.to_string(index=False))

print("\n=== RESOLUSI GAMBAR ===")
print(df.groupby(["width", "height"]).size().sort_values(ascending=False).head(10))

print("\n=== TOTAL ===")
print(f"Total gambar: {len(df)}")
print(f"Gambar rusak: {len(invalid_images)}")

if invalid_images:
    pd.DataFrame(invalid_images).to_csv(
        output_dir / "invalid_images.csv", index=False
    )

print("\nAudit awal selesai. Hasil disimpan di folder reports/")
