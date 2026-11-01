
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
MIN_DIMENSION = 320

# Nilai awal, perlu dikalibrasi dengan foto asli.
BLUR_REJECT = 25.0
BLUR_WARNING = 65.0


class ImageValidationError(Exception):
    def __init__(self, message, status_code=422):
        super().__init__(message)
        self.status_code = status_code


def decode_uploaded_image(content: bytes):
    if not content:
        raise ImageValidationError(
            "File gambar kosong.", 400
        )

    if len(content) > MAX_FILE_SIZE:
        raise ImageValidationError(
            "Ukuran gambar maksimal 10 MB.", 413
        )

    try:
        with Image.open(BytesIO(content)) as image:
            width, height = image.size

            if width * height > MAX_PIXELS:
                raise ImageValidationError(
                    "Resolusi gambar terlalu besar.", 413
                )

            image = ImageOps.exif_transpose(image)
            rgb = np.asarray(
                image.convert("RGB")
            ).copy()

    except ImageValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ):
        raise ImageValidationError(
            "File bukan gambar yang valid.", 400
        )

    return rgb


def calculate_image_quality(rgb):
    height, width = rgb.shape[:2]
    warnings = []

    if min(height, width) < MIN_DIMENSION:
        raise ImageValidationError(
            "Resolusi foto terlalu kecil. "
            "Gunakan foto minimal 320 piksel "
            "pada sisi terpendek."
        )

    # Samakan ukuran analisis agar metrik blur
    # lebih konsisten antara foto smartphone.
    scale = min(1.0, 640.0 / max(height, width))

    if scale < 1.0:
        sample = cv2.resize(
            rgb,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        sample = rgb

    gray = cv2.cvtColor(
        sample,
        cv2.COLOR_RGB2GRAY
    )

    brightness = float(
        np.mean(gray)
    )

    blur_score = float(
        cv2.Laplacian(
            gray, cv2.CV_64F
        ).var()
    )

    dark_fraction = float(
        np.mean(gray < 35)
    )

    bright_fraction = float(
        np.mean(gray > 225)
    )

    if blur_score < BLUR_REJECT:
        raise ImageValidationError(
            "Foto terdeteksi sangat buram. "
            "Ambil ulang dengan kamera stabil "
            "dan fokus pada telur."
        )

    if blur_score < BLUR_WARNING:
        warnings.append(
            "Foto mungkin kurang tajam. "
            "Hasil klasifikasi dapat terpengaruh."
        )

    # Tolak hanya bila hampir seluruh gambar
    # tidak mendapatkan pencahayaan memadai.
    if dark_fraction > 0.85:
        raise ImageValidationError(
            "Foto terlalu gelap. "
            "Ambil ulang dengan pencahayaan cukup."
        )

    if bright_fraction > 0.85:
        raise ImageValidationError(
            "Foto terlalu terang. "
            "Kurangi cahaya atau pantulan."
        )

    if dark_fraction > 0.50:
        warnings.append(
            "Sebagian besar gambar sangat gelap."
        )

    if bright_fraction > 0.50:
        warnings.append(
            "Sebagian besar gambar sangat terang."
        )

    return {
        "status": (
            "warning" if warnings else "passed"
        ),
        "width": int(width),
        "height": int(height),
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "dark_fraction": round(dark_fraction, 4),
        "bright_fraction": round(
            bright_fraction, 4
        ),
        "warnings": warnings
    }


def detect_egg_candidate(rgb):
    """
    Heuristik untuk satu objek oval pada
    background relatif seragam.

    Tidak dapat membuktikan objek adalah telur.
    Tidak menghapus background pada input CNN.
    """

    height, width = rgb.shape[:2]

    # Gambar analisis tetap 320x320 maksimum,
    # proporsi aslinya dipertahankan.
    scale = min(
        1.0,
        320.0 / max(height, width)
    )

    sample = cv2.resize(
        rgb,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    h, w = sample.shape[:2]

    # Ambil piksel sepanjang tepi sebagai
    # perkiraan background.
    border_size = max(
        5,
        int(min(h, w) * 0.05)
    )

    top = sample[:border_size, :, :]
    bottom = sample[-border_size:, :, :]
    left = sample[:, :border_size, :]
    right = sample[:, -border_size:, :]

    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0).astype(np.float32)

    background = np.median(
        border_pixels,
        axis=0
    )

    background_variation = float(
        np.mean(
            np.linalg.norm(
                border_pixels - background,
                axis=1
            )
        )
    )

    # Pada background kompleks, kontur ini
    # tidak dapat dipercaya sebagai deteksi telur.
    if background_variation > 35.0:
        return {
            "status": "unverified",
            "candidate_found": False,
            "reason": "complex_background",
            "background_variation": round(
                background_variation, 2
            ),
            "candidate_score": None
        }

    difference = np.linalg.norm(
        sample.astype(np.float32) - background,
        axis=2
    )

    difference = np.clip(
        difference, 0, 255
    ).astype(np.uint8)

    mask = np.where(
        difference > 35,
        255,
        0
    ).astype(np.uint8)

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        if len(contour) < 5:
            continue

        area = cv2.contourArea(contour)
        area_ratio = area / float(w * h)

        if not 0.05 <= area_ratio <= 0.85:
            continue

        (cx, cy), (major, minor), _ = (
            cv2.fitEllipse(contour)
        )

        longer = max(major, minor)
        shorter = min(major, minor)

        if shorter <= 0:
            continue

        aspect_ratio = longer / shorter

        if not 1.05 <= aspect_ratio <= 2.50:
            continue

        # Area ellipse teoretis.
        ellipse_area = (
            np.pi * major * minor / 4.0
        )

        if ellipse_area <= 0:
            continue

        ellipse_fill = area / ellipse_area

        if not 0.65 <= ellipse_fill <= 1.20:
            continue

        # Pastikan titik tengah objek tidak
        # terlalu dekat tepi foto.
        center_x = cx / w
        center_y = cy / h

        if not (
            0.12 <= center_x <= 0.88
            and 0.12 <= center_y <= 0.88
        ):
            continue

        # Indikator geometris saja, bukan
        # confidence model object detection.
        shape_score = 1.0 - min(
            abs(ellipse_fill - 1.0),
            1.0
        )

        candidates.append({
            "area_ratio": float(area_ratio),
            "aspect_ratio": float(aspect_ratio),
            "shape_score": float(shape_score)
        })

    if candidates:
        best = max(
            candidates,
            key=lambda item: item["area_ratio"]
        )

        return {
            "status": "candidate_detected",
            "candidate_found": True,
            "reason": "oval_contour_detected",
            "background_variation": round(
                background_variation, 2
            ),
            "candidate_score": round(
                best["shape_score"], 4
            ),
            "area_ratio": round(
                best["area_ratio"], 4
            ),
            "aspect_ratio": round(
                best["aspect_ratio"], 4
            ),
            "candidate_count": len(candidates)
        }

    return {
        "status": "unverified",
        "candidate_found": False,
        "reason": "no_oval_contour_detected",
        "background_variation": round(
            background_variation, 2
        ),
        "candidate_score": None
    }


def validate_egg_image(content: bytes):
    rgb = decode_uploaded_image(content)

    quality = calculate_image_quality(rgb)
    detection = detect_egg_candidate(rgb)

    warnings = list(quality["warnings"])

    if detection["status"] == "unverified":
        warnings.append(
            "Keberadaan telur belum dapat "
            "diverifikasi otomatis. Pastikan "
            "foto hanya menampilkan satu telur."
        )

    return {
        "image": rgb,
        "quality": quality,
        "detection": detection,
        "warnings": warnings
    }
