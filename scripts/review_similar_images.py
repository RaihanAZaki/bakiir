
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

REPORT_DIR = Path("reports")

df = pd.read_csv(
    REPORT_DIR / "near_duplicate_candidates.csv"
)

# Prioritaskan pHash 0 dan label berbeda
df = df.sort_values("distance")

different_label = df[
    df["label_a"] != df["label_b"]
]

same_label = df[
    df["label_a"] == df["label_b"]
]

# Ambil contoh dari dua kondisi
selected = pd.concat([
    different_label.head(12),
    same_label.head(12)
], ignore_index=True)

THUMB = (240, 200)
ROW_HEIGHT = 245
WIDTH = 520

sheet = Image.new(
    "RGB",
    (WIDTH, ROW_HEIGHT * len(selected)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, row in selected.iterrows():
    y = i * ROW_HEIGHT

    for j, col in enumerate(["image_a", "image_b"]):
        path = Path(row[col])

        with Image.open(path) as img:
            thumb = ImageOps.contain(
                img.convert("RGB"),
                THUMB
            )

        x = 10 + j * 260
        sheet.paste(thumb, (x, y + 5))

    draw.text(
        (10, y + 207),
        f"{row['label_a']} ({row['split_a']}) "
        f"vs {row['label_b']} ({row['split_b']})",
        fill="black"
    )

    draw.text(
        (10, y + 224),
        f"pHash distance: {row['distance']}",
        fill="black"
    )

output = REPORT_DIR / "similar_images_review.jpg"
sheet.save(output, quality=90)

print("Pemeriksaan selesai!")
print(f"Jumlah pasangan: {len(selected)}")
print(f"Hasil: {output}")
