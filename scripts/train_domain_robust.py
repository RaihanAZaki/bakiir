
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

# ============================================
# CONFIGURATION
# ============================================

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

EPOCHS = 25
LEARNING_RATE = 2e-6
FINE_TUNE_LAYERS = 50

CLASS_NAMES = [
    "Fresh",
    "Half-Fresh",
    "Rotten"
]

TRAIN_DIR = Path("dataset/cleaned/train")
VALID_DIR = Path("dataset/cleaned/valid")

BASE_MODEL_PATH = Path(
    "models/rotation_robust.keras"
)

OUTPUT_MODEL_PATH = Path(
    "models/domain_robust.keras"
)

REPORT_DIR = Path(
    "reports/model_h/training"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

tf.keras.utils.set_random_seed(SEED)


# ============================================
# DATA AUGMENTATION
# ============================================

rotation_layer = tf.keras.layers.RandomRotation(
    factor=0.5,
    fill_mode="constant",
    fill_value=0.0,
    seed=SEED
)

zoom_layer = tf.keras.layers.RandomZoom(
    height_factor=0.08,
    width_factor=0.08,
    fill_mode="constant",
    fill_value=0.0,
    seed=SEED + 1
)


def augment_one_image(image):
    image = tf.cast(image, tf.float32)

    # Sekitar 35% gambar tidak diubah.
    apply_augmentation = (
        tf.random.uniform([]) < 0.65
    )

    def transformed():
        result = image

        # Brightness: kurang lebih +/- 12
        # pada rentang nilai piksel 0-255.
        result = tf.image.random_brightness(
            result,
            max_delta=12.0
        )

        # Contrast
        result = tf.image.random_contrast(
            result,
            lower=0.85,
            upper=1.15
        )

        # Saturation
        result = tf.image.random_saturation(
            result,
            lower=0.90,
            upper=1.10
        )

        return tf.clip_by_value(
            result,
            0.0,
            255.0
        )

    return tf.cond(
        apply_augmentation,
        transformed,
        lambda: image
    )


def augment_batch(images, labels):
    images = tf.cast(
        images,
        tf.float32
    )

    # Rotasi dan zoom dilakukan pada
    # sebagian gambar, bukan semuanya.
    rotate_choice = (
        tf.random.uniform(
            (tf.shape(images)[0],)
        ) < 0.60
    )

    rotated = rotation_layer(
        images,
        training=True
    )

    images = tf.where(
        rotate_choice[:, None, None, None],
        rotated,
        images
    )

    zoom_choice = (
        tf.random.uniform(
            (tf.shape(images)[0],)
        ) < 0.40
    )

    zoomed = zoom_layer(
        images,
        training=True
    )

    images = tf.where(
        zoom_choice[:, None, None, None],
        zoomed,
        images
    )

    images = tf.map_fn(
        augment_one_image,
        images,
        fn_output_signature=tf.float32
    )

    return images, labels


# ============================================
# LOAD DATASET
# ============================================

def load_datasets():
    train_ds = (
        tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            labels="inferred",
            label_mode="int",
            class_names=CLASS_NAMES,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=SEED
        )
    )

    valid_ds = (
        tf.keras.utils.image_dataset_from_directory(
            VALID_DIR,
            labels="inferred",
            label_mode="int",
            class_names=CLASS_NAMES,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    )

    train_ds = train_ds.map(
        augment_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    valid_ds = valid_ds.prefetch(
        tf.data.AUTOTUNE
    )

    return train_ds, valid_ds


# ============================================
# LOAD MODEL G AND BUILD MODEL H
# ============================================

def build_model():
    if not BASE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model G tidak ditemukan: "
            f"{BASE_MODEL_PATH}"
        )

    if OUTPUT_MODEL_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_MODEL_PATH} sudah ada. "
            "Script dihentikan agar model "
            "sebelumnya tidak tertimpa."
        )

    print("\nLoading Model G...")

    model = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )

    backbone = None

    for layer in model.layers:
        if (
            isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        ):
            backbone = layer
            break

    if backbone is None:
        print("\nTop-level layers:")
        for layer in model.layers:
            print(
                layer.name,
                type(layer).__name__
            )

        raise ValueError(
            "Backbone MobileNetV2 tidak ditemukan."
        )

    # Aktifkan fine-tuning backbone.
    backbone.trainable = True

    # Bekukan semua layer backbone dahulu.
    for layer in backbone.layers:
        layer.trainable = False

    # Buka hanya 50 layer terakhir,
    # kecuali BatchNormalization.
    for layer in backbone.layers[
        -FINE_TUNE_LAYERS:
    ]:
        if not isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = True

    # Lapisan classifier milik Model G
    # tetap mengikuti status trainable-nya.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    trainable_params = sum(
        int(np.prod(w.shape))
        for w in model.trainable_weights
    )

    print("\n=== MODEL H CONFIGURATION ===")
    print("Backbone:", backbone.name)
    print(
        "Fine-tune last layers:",
        FINE_TUNE_LAYERS
    )
    print("Learning rate:", LEARNING_RATE)
    print(
        "Trainable parameters:",
        trainable_params
    )
    print("Base model:", BASE_MODEL_PATH)
    print("Output model:", OUTPUT_MODEL_PATH)

    return model


# ============================================
# SAVE TRAINING HISTORY
# ============================================

def save_history(history):
    history_data = {
        name: [float(x) for x in values]
        for name, values in history.history.items()
    }

    with open(
        REPORT_DIR / "history.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            history_data,
            file,
            indent=2
        )

    for metric in ["accuracy", "loss"]:
        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            history.history[metric],
            label=f"Train {metric}"
        )

        ax.plot(
            history.history[f"val_{metric}"],
            label=f"Validation {metric}"
        )

        ax.set_title(
            f"Model H - {metric}"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            REPORT_DIR / f"{metric}.png",
            dpi=150
        )

        plt.close(fig)


# ============================================
# TRAIN MODEL H
# ============================================

def main():
    print("\n=== MODEL H TRAINING ===")

    train_ds, valid_ds = load_datasets()
    model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    save_history(history)

    # Load checkpoint dengan val_loss terbaik.
    best_model = tf.keras.models.load_model(
        OUTPUT_MODEL_PATH,
        compile=False
    )

    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    val_loss, val_accuracy = (
        best_model.evaluate(
            valid_ds,
            verbose=1
        )
    )

    metrics = {
        "model": "domain_robust",
        "base_model": "rotation_robust",
        "validation_accuracy": float(
            val_accuracy
        ),
        "validation_loss": float(
            val_loss
        ),
        "epochs_completed": len(
            history.history["loss"]
        ),
        "learning_rate_initial": (
            LEARNING_RATE
        ),
        "fine_tune_layers": (
            FINE_TUNE_LAYERS
        ),
        "augmentation": [
            "rotation",
            "zoom",
            "brightness",
            "contrast",
            "saturation"
        ]
    }

    with open(
        REPORT_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2
        )

    print("\n=== MODEL H RESULTS ===")
    print(
        f"Validation accuracy: "
        f"{val_accuracy:.4%}"
    )
    print(
        f"Validation loss: "
        f"{val_loss:.4f}"
    )
    print(
        f"Epochs completed: "
        f"{metrics['epochs_completed']}"
    )
    print(
        f"Model saved: "
        f"{OUTPUT_MODEL_PATH}"
    )

    print("\nModel H training completed.")


if __name__ == "__main__":
    main()
