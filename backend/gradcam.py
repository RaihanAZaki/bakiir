
import base64
from threading import RLock

import cv2
import numpy as np
import tensorflow as tf


# ==========================================
# MODEL G GRAD-CAM
# ==========================================

class GradCamModel:
    def __init__(self, original_model):
        self.model = original_model
        self.lock = RLock()

        backbones = [
            layer
            for layer in original_model.layers
            if (
                isinstance(layer, tf.keras.Model)
                and "mobilenetv2" in layer.name.lower()
            )
        ]

        if not backbones:
            raise RuntimeError(
                "Backbone MobileNetV2 tidak ditemukan."
            )

        self.backbone = backbones[-1]
        self.layer_name = self.backbone.name + "/output"

    def forward_with_features(self, inputs):
        with self.lock:
            captured = []
            original_call = self.backbone.call

            def hooked_call(*args, **kwargs):
                output = original_call(*args, **kwargs)
                captured.append(output)
                return output

            try:
                self.backbone.call = hooked_call
                predictions = self.model(
                    inputs,
                    training=False,
                )
            finally:
                self.backbone.call = original_call

            if not captured:
                raise RuntimeError(
                    "Feature map MobileNetV2 tidak tertangkap."
                )

            features = captured[-1]

            if len(features.shape) != 4:
                raise RuntimeError(
                    "Feature map harus memiliki 4 dimensi."
                )

            return features, predictions


def build_gradcam_model(model):
    grad_model = GradCamModel(model)

    sample = tf.zeros(
        (1, 224, 224, 3),
        dtype=tf.float32,
    )

    features, predictions = (
        grad_model.forward_with_features(sample)
    )

    if predictions.shape[-1] != 3:
        raise RuntimeError(
            "Model G harus memiliki 3 kelas."
        )

    print(
        "[Grad-CAM] Feature map:",
        features.shape,
    )

    return grad_model, grad_model.layer_name
    

def compute_heatmap(
    grad_model,
    image,
    class_index,
):
    image = np.asarray(image, dtype=np.float32)

    if image.shape != (224, 224, 3):
        raise ValueError(
            f"Ukuran gambar tidak sesuai: {image.shape}"
        )

    tensor = tf.convert_to_tensor(
        image[None, ...],
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        features, predictions = (
            grad_model.forward_with_features(tensor)
        )

        score = predictions[:, class_index]

    gradients = tape.gradient(score, features)

    if gradients is None:
        raise RuntimeError(
            "Gradien tidak terhubung dengan prediksi."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    cam = tf.reduce_sum(
        features * weights[:, None, None, :],
        axis=-1,
    )

    cam = tf.nn.relu(cam)[0].numpy()

    return cv2.resize(
        cam.astype(np.float32),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def compute_tta_gradcam(
    grad_model,
    image,
    class_index,
    angles=(0, 90, 180, 270),
):
    heatmaps = []

    for angle in angles:
        if angle % 90 != 0:
            raise ValueError(
                "Sudut rotasi harus kelipatan 90."
            )

        k = (angle // 90) % 4

        rotated = np.rot90(image, k=k).copy()

        heatmap = compute_heatmap(
            grad_model,
            rotated,
            class_index,
        )

        restored = np.rot90(
            heatmap,
            k=-k,
        ).copy()

        heatmaps.append(restored)

    combined = np.mean(
        np.stack(heatmaps, axis=0),
        axis=0,
    )

    maximum = float(np.max(combined))

    if maximum > 1e-8:
        combined /= maximum
    else:
        combined = np.zeros_like(combined)

    return np.clip(combined, 0, 1)


# ==========================================
# EGG SEGMENTATION - NO YOLO
# ==========================================

def create_egg_mask(image):
    """
    Segmentasi telur menggunakan GrabCut.

    Asumsi:
    - Satu telur utama berada dekat tengah gambar.
    - Telur cukup berbeda dari background.

    Tidak cocok untuk seluruh kondisi gambar.
    """

    rgb = np.clip(
        image, 0, 255
    ).astype(np.uint8)

    height, width = rgb.shape[:2]

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    # Init GrabCut:
    # background dominan di area pinggir,
    # foreground di area tengah.
    gc_mask = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8,
    )

    # Border = background pasti.
    border = max(4, int(min(height, width) * 0.06))

    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    center = (width // 2, height // 2)

    # Ellipse luas = kemungkinan telur.
    cv2.ellipse(
        gc_mask,
        center,
        (
            int(width * 0.37),
            int(height * 0.34),
        ),
        0,
        0,
        360,
        cv2.GC_PR_FGD,
        -1,
    )

    # Pusat telur sebagai foreground pasti.
    cv2.ellipse(
        gc_mask,
        center,
        (
            int(width * 0.14),
            int(height * 0.13),
        ),
        0,
        0,
        360,
        cv2.GC_FGD,
        -1,
    )

    bg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )
    fg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    cv2.grabCut(
        bgr,
        gc_mask,
        None,
        bg_model,
        fg_model,
        6,
        cv2.GC_INIT_WITH_MASK,
    )

    binary = np.where(
        (gc_mask == cv2.GC_FGD)
        | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    # Hilangkan noise.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Ambil objek terbesar.
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError(
            "Segmentasi gagal: objek telur tidak ditemukan."
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    area_ratio = (
        cv2.contourArea(contour)
        / float(width * height)
    )

    # Hindari mask yang terlalu kecil/besar.
    if not 0.08 <= area_ratio <= 0.85:
        raise RuntimeError(
            "Segmentasi telur tidak meyakinkan: "
            f"area_ratio={area_ratio:.3f}"
        )

    result = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.drawContours(
        result,
        [contour],
        -1,
        255,
        thickness=-1,
    )

    if result[center[1], center[0]] == 0:
        raise RuntimeError(
            "Segmentasi tidak mencakup pusat telur."
        )

    # Haluskan tepi agar overlay natural.
    soft_mask = cv2.GaussianBlur(
        result.astype(np.float32) / 255.0,
        (7, 7),
        0,
    )

    return np.clip(soft_mask, 0, 1)


# ==========================================
# IMAGE ENCODING
# ==========================================

def encode_image(rgb):
    rgb = np.clip(
        rgb, 0, 255
    ).astype(np.uint8)

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    success, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        raise RuntimeError(
            "Gagal mengencode gambar."
        )

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")
    )


def make_overlay(
    image,
    heatmap,
    egg_mask=None,
    alpha=0.65,
):
    """
    Jika egg_mask diberikan, overlay hanya
    muncul pada area telur.

    Perhitungan CNN dan Grad-CAM asli
    tidak diubah oleh masking visual ini.
    """

    rgb = np.clip(
        image, 0, 255
    ).astype(np.uint8)

    height, width = rgb.shape[:2]

    heatmap = cv2.resize(
        heatmap.astype(np.float32),
        (width, height),
    )

    heatmap = np.clip(heatmap, 0, 1)

    colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB,
    )

    local_alpha = alpha * heatmap

    if egg_mask is not None:
        local_alpha *= egg_mask

    local_alpha = local_alpha[..., None]

    overlay = (
        rgb.astype(np.float32)
        * (1 - local_alpha)
        + colored.astype(np.float32)
        * local_alpha
    )

    return encode_image(overlay)


# ==========================================
# GENERATE EXPLANATION
# ==========================================

def generate_gradcam(
    grad_model,
    layer_name,
    image,
    class_index,
    angles=(0, 90, 180, 270),
):
    # Grad-CAM pada keseluruhan input model.
    heatmap = compute_tta_gradcam(
        grad_model,
        image,
        class_index,
        angles,
    )

    # Simpan visualisasi asli, tanpa masking.
    raw_image = make_overlay(
        image,
        heatmap,
    )

    # Segmentasi hanya untuk visualisasi.
    try:
        mask = create_egg_mask(image)

        focused_image = make_overlay(
            image,
            heatmap,
            egg_mask=mask,
        )

        segmentation_status = "success"
        message = (
            "Heatmap ditampilkan pada area telur. "
            "Mask hanya membatasi visualisasi; "
            "model tetap memproses seluruh gambar."
        )

    except Exception as exc:
        segmentation_status = "failed"

        # Jangan tampilkan mask yang tidak valid.
        focused_image = None

        message = (
            "Segmentasi area telur gagal: "
            + str(exc)
        )

    # Ukur lokasi Grad-CAM sebelum masking,
    # agar ketergantungan background terlihat.
    background_contribution = None

    if segmentation_status == "success":
        mass = float(heatmap.sum())

        if mass > 1e-8:
            background_contribution = float(
                (
                    heatmap * (1 - mask)
                ).sum() / mass
            )

    return {
        "available": focused_image is not None,
        "image": focused_image,
        "raw_image": raw_image,
        "layer": layer_name,
        "segmentation_status": segmentation_status,
        "background_contribution": background_contribution,
        "message": message,
    }
