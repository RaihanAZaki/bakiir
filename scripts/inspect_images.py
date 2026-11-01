
from pathlib import Path
from itertools import combinations
import random

import imagehash
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

DATASET_DIR = Path("dataset/cleaned")
REPORT_DIR = Path("reports")
CLASSES = ["Fresh", "Half-Fresh", "Rotten"]
SPLITS = ["train", "valid", "test"]

random.seed(42)
REPORT_DIR.mkdir(exist_ok=True)

records = []

# 1. Baca gambar dan hitung perceptual hash
for split in SPLITS:
    for label in CLASSES:
        folder = DATASET_DIR / split / label

        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            with Image.open(path) as image:
                phash = imagehash.phash(image.convert("RGB"))

            records.append({
                "split": split,
                "label": label,
                "path": str(path),
                "phash": phash,
            })

print(f"Total gambar diperiksa: {len(records)}")

# 2. Cari pasangan gambar mirip lintas split
# Threshold <= 5 hanya untuk kandidat pemeriksaan manual
suspects = []

for a, b in combinations(records, 2):
    if a["split"] == b["split"]:
        continue

    distance = a["phash"] - b["phash"]

    if distance <= 5:
        suspects.append({
            "image_a": a["path"],
            "split_a": a["split"],
            "label_a": a["label"],
            "image_b": b["path"],
            "split_b": b["split"],
            "label_b": b["label"],
            "distance": distance,
        })

suspects.sort(key=lambda x: x["distance"])

pd.DataFrame(
    suspects,
    columns=[
        "image_a", "split_a", "label_a",
        "image_b", "split_b", "label_b", "distance"
    ]
).to_csv(
    REPORT_DIR / "near_duplicate_candidates.csv",
    index=False
)

# 3. Buat contact sheet sebanyak 9 gambar per kelas
sample_count = 9
thumb_size = (150, 150)
cell_width = 180
cell_height = 195

sheet = Image.new(
    "RGB",
    (cell_width * sample_count, cell_height * len(CLASSES)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for row_index, label in enumerate(CLASSES):
    candidates = [
        r for r in records
        if r["label"] == label and r["split"] == "train"
    ]

    selected = random.sample(
        candidates,
        min(sample_count, len(candidates))
    )

    for col_index, item in enumerate(selected):
        with Image.open(item["path"]) as img:
            thumbnail = ImageOps.contain(
                img.convert("RGB"),
                thumb_size
            )

        x = col_index * cell_width
        y = row_index * cell_height

        sheet.paste(thumbnail, (x + 10, y + 5))

        draw.text(
            (x + 10, y + 160),
            label,
            fill="black"
        )

        draw.text(
            (x + 10, y + 176),
            Path(item["path"]).stem[:22],
            fill="black"
        )

contact_path = REPORT_DIR / "sample_contact_sheet.jpg"
sheet.save(contact_path, quality=95)

print("\n=== HASIL PEMERIKSAAN ===")
print(f"Total gambar: {len(records)}")
print(f"Kandidat near-duplicate lintas split: {len(suspects)}")
print(f"Contact sheet: {contact_path}")
print("Laporan: reports/near_duplicate_candidates.csv")
print("\nPemeriksaan selesai.")
