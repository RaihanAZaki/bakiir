
from pathlib import Path
import csv

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_CSV = Path(
    "reports/dataset_audit/visual_review_pairs.csv"
)

OUTPUT_DIR = Path(
    "reports/dataset_audit/verification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIORITY_PAIRS = [1, 2, 3, 6, 8, 9, 10, 14, 16]

IMAGE_SIZE = (256, 256)
PANEL_SIZE = 256

# ==========================================
# LOAD IMAGE
# ==========================================

def load_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB")).copy()


def resize_image(image):
    return cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )


# ==========================================
# METRICS
# ==========================================

def calculate_mae(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    return float(np.mean(np.abs(a - b)))


def calculate_ssim(a, b):
    # SSIM grayscale, dihitung pada piksel
    # yang telah di-resize ke ukuran sama.
    gray_a = cv2.cvtColor(
        a, cv2.COLOR_RGB2GRAY
    ).astype(np.float64)

    gray_b = cv2.cvtColor(
        b, cv2.COLOR_RGB2GRAY
    ).astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(
        gray_a, (11, 11), 1.5
    )

    mu_b = cv2.GaussianBlur(
        gray_b, (11, 11), 1.5
    )

    sigma_a = cv2.GaussianBlur(
        gray_a ** 2, (11, 11), 1.5
    ) - mu_a ** 2

    sigma_b = cv2.GaussianBlur(
        gray_b ** 2, (11, 11), 1.5
    ) - mu_b ** 2

    sigma_ab = cv2.GaussianBlur(
        gray_a * gray_b, (11, 11), 1.5
    ) - mu_a * mu_b

    numerator = (
        (2 * mu_a * mu_b + c1)
        * (2 * sigma_ab + c2)
    )

    denominator = (
        (mu_a ** 2 + mu_b ** 2 + c1)
        * (sigma_a + sigma_b + c2)
    )

    return float(
        np.mean(numerator / (
            denominator + 1e-12
        ))
    )


# ==========================================
# DIFFERENCE MAP
# ==========================================

def create_difference(a, b):
    difference = cv2.absdiff(a, b)

    gray = cv2.cvtColor(
        difference,
        cv2.COLOR_RGB2GRAY
    )

    # Bright = area yang berbeda.
    # Dark = area yang mirip.
    normalized = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    heatmap_bgr = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB
    )

    return heatmap


# ==========================================
# VISUAL REPORT
# ==========================================

def create_panel(
    pair_number,
    image_a,
    image_b,
    difference,
    mae,
    ssim,
    row
):
    width = PANEL_SIZE * 3
    height = PANEL_SIZE + 85

    panel = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(panel)

    images = [
        Image.fromarray(image_a),
        Image.fromarray(image_b),
        Image.fromarray(difference)
    ]

    titles = [
        "Image A",
        "Image B",
        "Difference Map"
    ]

    for i, img in enumerate(images):
        x = i * PANEL_SIZE

        panel.paste(img, (x, 25))
        draw.text(
            (x + 8, 6),
            titles[i],
            fill="black"
        )

    description = (
        f"Pair #{pair_number} | "
        f"pHash: {row['phash_distance']} | "
        f"SSIM: {ssim:.4f} | MAE: {mae:.2f}"
    )

    draw.text(
        (10, PANEL_SIZE + 35),
        description,
        fill="black"
    )

    classes = (
        f"{row['split_a']}/{row['class_a']} "
        f"vs {row['split_b']}/{row['class_b']}"
    )

    draw.text(
        (10, PANEL_SIZE + 57),
        classes,
        fill="black"
    )

    return panel


# ==========================================
# MAIN
# ==========================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_CSV}"
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

    records = []
    panels = []

    print("\n=== VERIFY NEAR DUPLICATES ===")

    for pair_number in PRIORITY_PAIRS:
        index = pair_number - 1

        if index >= len(df):
            print(
                f"Skip pair #{pair_number}: tidak tersedia"
            )
            continue

        row = df.iloc[index]

        path_a = Path(row["path_a"])
        path_b = Path(row["path_b"])

        if not path_a.is_file():
            raise FileNotFoundError(path_a)

        if not path_b.is_file():
            raise FileNotFoundError(path_b)

        original_a = load_image(path_a)
        original_b = load_image(path_b)

        image_a = resize_image(original_a)
        image_b = resize_image(original_b)

        mae = calculate_mae(image_a, image_b)
        ssim = calculate_ssim(image_a, image_b)

        difference = create_difference(
            image_a,
            image_b
        )

        records.append({
            "pair_number": pair_number,
            "split_a": row["split_a"],
            "split_b": row["split_b"],
            "class_a": row["class_a"],
            "class_b": row["class_b"],
            "phash_distance": int(
                row["phash_distance"]
            ),
            "ssim": ssim,
            "mae": mae,
            "path_a": str(path_a),
            "path_b": str(path_b)
        })

        panel = create_panel(
            pair_number,
            image_a,
            image_b,
            difference,
            mae,
            ssim,
            row
        )

        panel_path = (
            OUTPUT_DIR /
            f"pair_{pair_number:02d}.jpg"
        )

        panel.save(panel_path, quality=95)
        panels.append(panel)

        print(
            f"Pair #{pair_number:02d} | "
            f"SSIM={ssim:.4f} | MAE={mae:.2f}"
        )

    if not panels:
        raise RuntimeError(
            "Tidak ada pasangan yang berhasil diproses."
        )

    # ======================================
    # SAVE METRICS
    # ======================================

    metrics_path = (
        OUTPUT_DIR / "verification_metrics.csv"
    )

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    # ======================================
    # COMBINED CONTACT SHEET
    # ======================================

    sheet_width = PANEL_SIZE * 3
    sheet_height = sum(
        panel.height for panel in panels
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    y = 0

    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height

    sheet_path = (
        OUTPUT_DIR / "verification_contact_sheet.jpg"
    )

    sheet.save(sheet_path, quality=95)

    print("\n=== VERIFICATION FINISHED ===")
    print("Pairs checked:", len(records))
    print("Metrics:", metrics_path)
    print("Contact sheet:", sheet_path)


if __name__ == "__main__":
    main()
