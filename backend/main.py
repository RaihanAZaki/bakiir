
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .image_validator import (
    ImageValidationError,
    validate_egg_image,
)

from .gradcam import (
    build_gradcam_model,
    generate_gradcam,
)


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "rotation_robust.keras"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Fresh",
    "Half-Fresh",
    "Rotten",
]

TTA_ANGLES = [0, 90, 180, 270]

MAX_FILE_SIZE = 10 * 1024 * 1024

model_lock = Lock()


# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="EggVision API",
    description=(
        "Egg Freshness Classification "
        "using Model G and Grad-CAM"
    ),
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ==========================================
# LOAD MODEL G
# ==========================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model G tidak ditemukan: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

if model.output_shape[-1] != 3:
    raise RuntimeError(
        "Model G harus menghasilkan 3 output."
    )

print("[EggVision] Model G loaded:", MODEL_PATH)


# ==========================================
# INITIALIZE GRAD-CAM
# ==========================================

gradcam_model = None
gradcam_layer = None
gradcam_error = None

try:
    gradcam_model, gradcam_layer = (
        build_gradcam_model(model)
    )

    print(
        "[EggVision] Grad-CAM layer:",
        gradcam_layer,
    )

except Exception as exc:
    gradcam_error = str(exc)

    print(
        "[EggVision] Grad-CAM initialization failed:",
        repr(exc),
    )


# ==========================================
# PREPROCESSING
# ==========================================

def preprocess_image(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Model G memiliki preprocessing internal.

    Input RGB 0..255.
    Tidak menggunakan preprocessing MobileNetV2
    secara eksternal.
    """

    image = tf.image.resize(
        rgb,
        IMAGE_SIZE,
        method="bilinear",
    )

    return tf.cast(
        image,
        tf.float32,
    ).numpy()


# ==========================================
# MODEL G + TTA
# ==========================================

def predict_with_tta(
    image: np.ndarray,
) -> dict:

    variants = np.stack(
        [
            np.rot90(
                image,
                k=angle // 90,
            ).copy()
            for angle in TTA_ANGLES
        ],
        axis=0,
    ).astype(np.float32)

    with model_lock:
        outputs = model(
            variants,
            training=False,
        ).numpy()

    if outputs.shape != (4, 3):
        raise RuntimeError(
            f"Output Model G tidak sesuai: {outputs.shape}"
        )

    probabilities = outputs.mean(axis=0)

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(
            "Probabilitas model tidak valid."
        )

    class_index = int(
        np.argmax(probabilities)
    )

    return {
        "prediction": CLASS_NAMES[class_index],
        "confidence": float(
            probabilities[class_index]
        ),
        "probabilities": {
            "Fresh": float(probabilities[0]),
            "Half-Fresh": float(probabilities[1]),
            "Rotten": float(probabilities[2]),
        },
        "model": "rotation_robust",
        "inference_method": "tta_4_angles",
        "tta_angles": TTA_ANGLES,
    }


# ==========================================
# GRAD-CAM
# ==========================================

def explain_prediction(
    image: np.ndarray,
    prediction: str,
) -> dict:

    if gradcam_model is None:
        return {
            "available": False,
            "image": None,
            "layer": None,
            "message": (
                "Grad-CAM belum tersedia. "
                + (gradcam_error or "")
            ),
        }

    class_index = CLASS_NAMES.index(
        prediction
    )

    with model_lock:
        explanation = generate_gradcam(
            grad_model=gradcam_model,
            layer_name=gradcam_layer,
            image=image,
            class_index=class_index,
            angles=TTA_ANGLES,
        )

    return explanation


# ==========================================
# UPLOAD FILE
# ==========================================

async def read_upload(
    file: UploadFile,
) -> bytes:

    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Gunakan JPG, PNG, atau WebP.",
        )

    content = await file.read(
        MAX_FILE_SIZE + 1
    )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="File gambar kosong.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Ukuran gambar maksimal 10 MB.",
        )

    return content


def validate_image(
    content: bytes,
):
    try:
        return validate_egg_image(
            content
        )

    except ImageValidationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc


# ==========================================
# BASIC ROUTES
# ==========================================

@app.get("/")
def root():
    return {
        "application": "EggVision",
        "status": "running",
        "model": "rotation_robust",
        "classes": CLASS_NAMES,
        "tta_angles": TTA_ANGLES,
        "gradcam_available": gradcam_model is not None,
        "object_detection": False,
        "database": False,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "gradcam_available": gradcam_model is not None,
        "gradcam_layer": gradcam_layer,
        "gradcam_error": gradcam_error,
    }


# ==========================================
# PREDICT
# ==========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):

    content = await read_upload(
        file
    )

    validation = await run_in_threadpool(
        validate_image,
        content,
    )

    rgb = validation["image"]

    image = preprocess_image(
        rgb
    )

    # --------------------------------------
    # CLASSIFICATION
    # --------------------------------------

    try:
        result = await run_in_threadpool(
            predict_with_tta,
            image,
        )

    except Exception as exc:
        print(
            "[EggVision] Prediction error:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Gagal melakukan klasifikasi.",
        ) from exc

    # --------------------------------------
    # GRAD-CAM
    # --------------------------------------

    try:
        explanation = await run_in_threadpool(
            explain_prediction,
            image,
            result["prediction"],
        )

    except Exception as exc:
        print(
            "[EggVision] Grad-CAM error:",
            repr(exc),
        )

        explanation = {
            "available": False,
            "image": None,
            "layer": gradcam_layer,
            "message": (
                "Gagal menghasilkan visualisasi Grad-CAM."
            ),
        }

    # --------------------------------------
    # RESPONSE
    # --------------------------------------

    result.update({
        "image_quality": validation["quality"],
        "egg_detection": validation["detection"],
        "warnings": validation["warnings"],
        "explanation": explanation,
    })

    return result
