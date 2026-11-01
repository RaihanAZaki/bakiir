
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, ImageDraw

INPUT_CSV = Path(
    "reports/dataset_audit/near_duplicate_candidates.csv"
)

OUTPUT_DIR = Path("reports/dataset_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLES_PER_GROUP = 5

CELL_W = 250
CELL_H = 245
PAGE_ROWS = 10


def load_thumbnail(path):
    with Image.open(path) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")
        return ImageOps.contain(image, (220, 175))


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"CSV tidak ditemukan: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required = {
        "path_a",
        "path_b",
        "split_a",
        "split_b",
        "class_a",
        "class_b",
        "phash_distance"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Kolom CSV tidak sesuai: {list(df.columns)}"
        )

    df["same_label"] = (
        df["class_a"] == df["class_b"]
    )

    samples = []

    for distance in [0, 2, 4]:
        for same_label in [True, False]:
            group = df[
                (df["phash_distance"] == distance)
                & (df["same_label"] == same_label)
            ]

            if group.empty:
                continue

            chosen = group.sample(
                n=min(SAMPLES_PER_GROUP, len(group)),
                random_state=SEED
            )

            samples.append(chosen)

    if not samples:
        raise ValueError(
            "Tidak ada kandidat untuk ditinjau."
        )

    selected = pd.concat(
        samples,
        ignore_index=True
    )

    # Simpan informasi pasangan yang ditinjau.
    selected.to_csv(
        OUTPUT_DIR / "visual_review_pairs.csv",
        index=False
    )

    print(
        f"Jumlah pasangan terpilih: {len(selected)}"
    )

    # Buat contact sheet.
    sheet = Image.new(
        "RGB",
        (
            CELL_W * 2,
            CELL_H * len(selected)
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, row in selected.iterrows():
        paths = [
            Path(row["path_a"]),
            Path(row["path_b"])
        ]

        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(
                f"File tidak ditemukan pada pasangan "
                f"{index + 1}: {paths}"
            )

        y = index * CELL_H

        for column, path in enumerate(paths):
            thumbnail = load_thumbnail(path)
            x = column * CELL_W + 10

            sheet.paste(
                thumbnail,
                (x, y + 8)
            )

        label = (
            f"#{index + 1} | "
            f"pHash={row['phash_distance']} | "
            f"{row['split_a']}/{row['class_a']} vs "
            f"{row['split_b']}/{row['class_b']}"
        )

        draw.text(
            (8, y + 192),
            label[:78],
            fill="black"
        )

        draw.line(
            [
                (0, y + CELL_H - 1),
                (CELL_W * 2, y + CELL_H - 1)
            ],
            fill="#cccccc",
            width=1
        )

    # Bagi contact sheet menjadi beberapa halaman.
    for start in range(
        0,
        len(selected),
        PAGE_ROWS
    ):
        end = min(
            start + PAGE_ROWS,
            len(selected)
        )

        page = sheet.crop(
            (
                0,
                start * CELL_H,
                CELL_W * 2,
                end * CELL_H
            )
        )

        page_no = start // PAGE_ROWS + 1

        output_path = (
            OUTPUT_DIR /
            f"visual_review_{page_no}.jpg"
        )

        page.save(
            output_path,
            quality=93
        )

        print(f"Saved: {output_path}")

    print("\n=== VISUAL REVIEW SELESAI ===")
    print(
        f"Total pasangan: {len(selected)}"
    )
    print(
        f"Output folder: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
