
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageOps, ImageDraw

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = Path("models/rotation_robust.keras")
IMAGE_DIR = Path("dataset/external_test/egg_001")
REPORT_DIR = Path("reports/model_h")

CLASS_NAMES = ["Fresh", "Half-Fresh", "Rotten"]
ANGLES = [0, 90, 180, 270]
IMG_SIZE = (224, 224)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(IMAGE_DIR)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# ==========================================
# LOAD IMAGE
# ==========================================

def load_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.asarray(
            img.convert("RGB")
        ).copy()


# ==========================================
# BACKGROUND SEGMENTATION
# ==========================================

def segment_egg(rgb):
    height, width = rgb.shape[:2]

    # Perkiraan awal: telur berada di area tengah.
    # Bagian tepi dianggap background.
    mask = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    border = max(2, int(min(height, width) * 0.03))

    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    # Area tengah sebagai kandidat foreground.
    center = (width // 2, height // 2)
    axes = (
        max(1, int(width * 0.38)),
        max(1, int(height * 0.38))
    )

    cv2.ellipse(
        mask,
        center,
        axes,
        0,
        0,
        360,
        cv2.GC_PR_FGD,
        -1
    )

    cv2.ellipse(
        mask,
        center,
        (
            max(1, int(width * 0.10)),
            max(1, int(height * 0.10))
        ),
        0,
        0,
        360,
        cv2.GC_FGD,
        -1
    )

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    cv2.grabCut(
        bgr,
        mask,
        None,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    binary = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Ambil komponen foreground terbesar.
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if count <= 1:
        raise ValueError(
            "Objek telur tidak terdeteksi"
        )

    largest = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    binary = np.where(
        labels == largest,
        255,
        0
    ).astype(np.uint8)

    area_ratio = np.count_nonzero(binary) / binary.size

    if not 0.03 <= area_ratio <= 0.90:
        raise ValueError(
            f"Mask tidak wajar: {area_ratio:.2%}"
        )

    # Terapkan mask tanpa mengubah warna telur.
    segmented = np.zeros_like(rgb)
    segmented[binary > 0] = rgb[binary > 0]

    return segmented, binary


# ==========================================
# PREPROCESSING
# ==========================================

def prepare_image(rgb):
    resized = tf.image.resize(
        rgb,
        IMG_SIZE,
        method="bilinear"
    )

    return resized.numpy().astype(np.float32)


def predict_original(rgb):
    image = prepare_image(rgb)

    probs = model.predict(
        image[None, ...],
        verbose=0
    )

    return probs[0]


def predict_tta(rgb):
    image = prepare_image(rgb)

    variants = [
        np.rot90(image, k=k)
        for k in range(4)
    ]

    batch = np.stack(
        variants,
        axis=0
    ).astype(np.float32)

    probs = model.predict(
        batch,
        verbose=0
    )

    return probs.mean(axis=0)


# ==========================================
# PREDICTION
# ==========================================

def predict_label(probs):
    index = int(np.argmax(probs))

    return (
        CLASS_NAMES[index],
        float(probs[index] * 100)
    )


def main():
    paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in (
            ".jpg", ".jpeg", ".png"
        )
    )

    if not paths:
        raise ValueError(
            "Tidak ditemukan gambar pengujian"
        )

    records = []
    previews = []

    for path in paths:
        print(f"\nMemproses: {path.name}")

        original = load_image(path)

        segmented, binary_mask = segment_egg(
            original
        )

        methods = {
            "original": predict_original(original),
            "segmentation": predict_original(segmented),
            "segmentation_tta": predict_tta(segmented)
        }

        for method, probs in methods.items():
            label, confidence = predict_label(probs)

            record = {
                "filename": path.name,
                "method": method,
                "prediction": label,
                "confidence": confidence,
            }

            for i, name in enumerate(CLASS_NAMES):
                record[f"prob_{name}"] = (
                    float(probs[i] * 100)
                )

            records.append(record)

            print(
                f"{method}: "
                f"{label} ({confidence:.2f}%)"
            )

        previews.append(
            (path.name, original, binary_mask, segmented)
        )

    # Simpan seluruh prediksi
    df = pd.DataFrame(records)

    df.to_csv(
        REPORT_DIR / "segmentation_predictions.csv",
        index=False
    )

    # Konsistensi antarfoto telur yang sama
    summary = []

    for method, group in df.groupby("method"):
        majority = (
            group["prediction"].value_counts().max()
        )

        summary.append({
            "method": method,
            "total_photos": len(group),
            "majority_agreement": majority / len(group),
            "distinct_predictions": group[
                "prediction"
            ].nunique()
        })

    pd.DataFrame(summary).to_csv(
        REPORT_DIR / "segmentation_consistency.csv",
        index=False
    )

    # ======================================
    # VISUAL PREVIEW
    # ======================================

    cell_size = 240
    heading_height = 50
    width = cell_size * 3
    height = heading_height + len(previews) * cell_size

    sheet = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, title in enumerate(
        ["Original", "Mask", "Segmented"]
    ):
        draw.text(
            (i * cell_size + 10, 15),
            title,
            fill="black"
        )

    for row, (filename, original, mask, segmented) in enumerate(previews):
        y = heading_height + row * cell_size

        images = [
            Image.fromarray(original),
            Image.fromarray(mask).convert("RGB"),
            Image.fromarray(segmented)
        ]

        for column, img in enumerate(images):
            thumb = ImageOps.contain(
                img,
                (220, 200)
            )

            x = column * cell_size + 10

            sheet.paste(
                thumb,
                (x, y + 10)
            )

        draw.text(
            (10, y + 210),
            filename,
            fill="black"
        )

    sheet.save(
        REPORT_DIR / "segmented_preview.jpg",
        quality=95
    )

    print("\n=== EKSPERIMEN SELESAI ===")
    print("Laporan: reports/model_h/")
    print("\nKonsistensi:")
    print(
        pd.DataFrame(summary).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
