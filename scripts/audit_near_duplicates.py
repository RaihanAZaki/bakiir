
from pathlib import Path
import csv
import itertools

import cv2
import numpy as np
import pandas as pd

MANIFEST_PATH = Path(
    "reports/cleaned_dataset_manifest.csv"
)

OUTPUT_DIR = Path("reports/dataset_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hamming distance <= 5 hanya digunakan
# sebagai filter kandidat, bukan bukti duplikat.
MAX_DISTANCE = 5


def calculate_phash(path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(path)

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        (32, 32),
        interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    dct = cv2.dct(resized)

    values = dct[:8, :8].copy()
    median = np.median(values.flatten()[1:])

    bits = (values > median).flatten()

    return sum(
        int(bit) << i
        for i, bit in enumerate(bits)
    )


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required = {
        "split", "freshness",
        "chicken_type", "output_path"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Kolom manifest tidak lengkap"
        )

    images = []

    print("Menghitung perceptual hash...")

    for row in df.itertuples(index=False):
        path = Path(row.output_path)

        if not path.exists():
            raise FileNotFoundError(path)

        images.append({
            "split": row.split,
            "freshness": row.freshness,
            "chicken_type": row.chicken_type,
            "path": str(path),
            "phash": calculate_phash(path)
        })

    candidates = []

    print("Memeriksa kemiripan antarsplit...")

    for a, b in itertools.combinations(images, 2):
        if a["split"] == b["split"]:
            continue

        # Batasi pada jenis ayam yang sama.
        if a["chicken_type"] != b["chicken_type"]:
            continue

        distance = (
            a["phash"] ^ b["phash"]
        ).bit_count()

        if distance > MAX_DISTANCE:
            continue

        candidates.append({
            "split_a": a["split"],
            "split_b": b["split"],
            "class_a": a["freshness"],
            "class_b": b["freshness"],
            "chicken_type": a["chicken_type"],
            "phash_distance": distance,
            "path_a": a["path"],
            "path_b": b["path"]
        })

    candidates.sort(
        key=lambda x: x["phash_distance"]
    )

    output = (
        OUTPUT_DIR / "near_duplicate_candidates.csv"
    )

    columns = [
        "split_a", "split_b",
        "class_a", "class_b",
        "chicken_type",
        "phash_distance", "path_a", "path_b"
    ]

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(candidates)

    print("\n=== AUDIT SELESAI ===")
    print("Gambar:", len(images))
    print("Kandidat pasangan:", len(candidates))
    print("Laporan:", output)
    print(
        "\nCatatan: kandidat belum tentu "
        "merupakan telur fisik yang sama."
    )


if __name__ == "__main__":
    main()
