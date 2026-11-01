
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocessing import load_dataset

# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
EPOCHS = 20
LEARNING_RATE = 2e-6
FINE_TUNE_LAYERS = 50

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

INPUT_MODEL = MODEL_DIR / "finetuning_advanced.keras"
OUTPUT_MODEL = MODEL_DIR / "rotation_robust.keras"

HISTORY_PATH = REPORT_DIR / "rotation_robust_history.json"
METRICS_PATH = REPORT_DIR / "rotation_robust_metrics.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# ==========================================
# ROTATION CONFIGURATION
# ==========================================

# 0 = tetap asli
# 1-5 = rotasi tetap
# 6 = rotasi acak

ROTATION_ANGLES = [45, 90, 135, 180, 270]

fixed_rotation_layers = [
    tf.keras.layers.RandomRotation(
        factor=(angle / 360.0, angle / 360.0),
        fill_mode="constant",
        fill_value=0.0,
        seed=SEED + i
    )
    for i, angle in enumerate(ROTATION_ANGLES)
]

random_rotation = tf.keras.layers.RandomRotation(
    factor=0.5,
    fill_mode="constant",
    fill_value=0.0,
    seed=SEED + 100
)


def augment_batch(images, labels):
    batch_size = tf.shape(images)[0]

    # Distribusi:
    # 40% gambar asli
    # 10% masing-masing untuk 6 jenis rotasi
    probability = tf.random.uniform(
        shape=(batch_size,),
        minval=0,
        maxval=1,
        dtype=tf.float32
    )

    # 0: asli
    # 1-5: rotasi tetap
    # 6: rotasi acak
    choices = tf.where(
        probability < 0.4,
        tf.zeros_like(probability, dtype=tf.int32),
        tf.minimum(
            tf.cast(
                (probability - 0.4) / 0.1,
                tf.int32
            ) + 1,
            6
        )
    )

    # Mulai dari gambar asli
    augmented = images

    # Terapkan rotasi tetap
    for i, layer in enumerate(fixed_rotation_layers):
        rotated = layer(images, training=True)

        mask = tf.equal(choices, i + 1)
        mask = mask[:, None, None, None]

        augmented = tf.where(
            mask,
            rotated,
            augmented
        )

    # Rotasi acak
    randomly_rotated = random_rotation(
        images,
        training=True
    )

    random_mask = tf.equal(choices, 6)
    random_mask = random_mask[:, None, None, None]

    augmented = tf.where(
        random_mask,
        randomly_rotated,
        augmented
    )

    return augmented, labels


# ==========================================
# DATASET
# ==========================================

def create_datasets():
    train_ds = load_dataset(
        "train",
        shuffle=True
    )

    valid_ds = load_dataset(
        "valid",
        shuffle=False
    )

    train_ds = train_ds.map(
        augment_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    return train_ds, valid_ds


# ==========================================
# MODEL G
# ==========================================

def build_model():
    if not INPUT_MODEL.exists():
        raise FileNotFoundError(
            f"Model F tidak ditemukan: {INPUT_MODEL}"
        )

    if OUTPUT_MODEL.exists():
        raise FileExistsError(
            f"{OUTPUT_MODEL} sudah ada. "
            "Pindahkan hasil Model G lama terlebih dahulu "
            "agar tidak tertimpa."
        )

    print("\nLoading Model F...")

    model = tf.keras.models.load_model(
        INPUT_MODEL,
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
        raise ValueError(
            "Backbone MobileNetV2 tidak ditemukan. "
            "Periksa struktur model."
        )

    backbone.trainable = True

    # Bekukan layer awal
    for layer in backbone.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    # BatchNormalization tetap frozen
    for layer in backbone.layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    trainable_params = sum(
        int(tf.size(w))
        for w in model.trainable_weights
    )

    print("\n=== MODEL G CONFIGURATION ===")
    print("Backbone:", backbone.name)
    print("Fine-tune layers:", FINE_TUNE_LAYERS)
    print("Learning rate:", LEARNING_RATE)
    print("Trainable parameters:", trainable_params)
    print("Input model:", INPUT_MODEL)
    print("Output model:", OUTPUT_MODEL)

    return model


# ==========================================
# TRAINING GRAPH
# ==========================================

def save_graphs(history):
    for metric in ["accuracy", "loss"]:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            history.history[metric],
            label=f"Training {metric}"
        )

        ax.plot(
            history.history[f"val_{metric}"],
            label=f"Validation {metric}"
        )

        ax.set_title(f"Rotation Robust - {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            REPORT_DIR / f"rotation_robust_{metric}.png",
            dpi=150
        )

        plt.close(fig)


# ==========================================
# MAIN TRAINING
# ==========================================

def main():
    print("\n=== TRAINING MODEL G ===")

    train_ds, valid_ds = create_datasets()
    model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_MODEL),
            monitor="val_loss",
            mode="min",
            save_best_only=True
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    save_graphs(history)

    history_data = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }

    with open(HISTORY_PATH, "w") as file:
        json.dump(history_data, file, indent=2)

    # Evaluasi checkpoint terbaik
    best_model = tf.keras.models.load_model(
        OUTPUT_MODEL,
        compile=False
    )

    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    val_loss, val_accuracy = best_model.evaluate(
        valid_ds,
        verbose=1
    )

    metrics = {
        "model": "rotation_robust",
        "base_model": "finetuning_advanced",
        "validation_accuracy": float(val_accuracy),
        "validation_loss": float(val_loss),
        "epochs_completed": len(history.history["loss"]),
        "learning_rate_initial": LEARNING_RATE,
        "fine_tune_layers": FINE_TUNE_LAYERS
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=2)

    print("\n=== HASIL MODEL G ===")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Model tersimpan: {OUTPUT_MODEL}")

    print("\nTraining Model G selesai!")


if __name__ == "__main__":
    main()
