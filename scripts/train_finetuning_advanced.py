
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
EPOCHS = 25
LEARNING_RATE = 5e-6
FINE_TUNE_LAYERS = 50
ROTATION_PROBABILITY = 0.60

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

INPUT_MODEL = MODEL_DIR / "finetuning.keras"
OUTPUT_MODEL = MODEL_DIR / "finetuning_advanced.keras"

tf.keras.utils.set_random_seed(SEED)

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ==========================================
# ROTATION AUGMENTATION
# ==========================================

rotation_layer = tf.keras.layers.RandomRotation(
    factor=0.5,
    fill_mode="constant",
    fill_value=0.0,
    seed=SEED,
)


def augment_batch(images, labels):
    rotated = rotation_layer(images, training=True)

    # Setiap gambar berpeluang 60% untuk dirotasi
    mask = tf.random.uniform(
        shape=(tf.shape(images)[0], 1, 1, 1)
    ) < ROTATION_PROBABILITY

    augmented = tf.where(mask, rotated, images)

    return augmented, labels


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

    return train_ds.prefetch(tf.data.AUTOTUNE), valid_ds


# ==========================================
# BUILD FINETUNING MODEL
# ==========================================

def build_model():
    print("\nLoading Model E...")

    if not INPUT_MODEL.exists():
        raise FileNotFoundError(INPUT_MODEL)

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
            "Backbone MobileNetV2 tidak ditemukan."
        )

    backbone.trainable = True

    for layer in backbone.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    # Batch Normalization tetap dibekukan
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

    print("\n=== MODEL F CONFIGURATION ===")
    print("Backbone:", backbone.name)
    print("Fine-tune layers:", FINE_TUNE_LAYERS)
    print("Learning rate:", LEARNING_RATE)
    print("Rotation probability:", ROTATION_PROBABILITY)

    return model


# ==========================================
# SAVE TRAINING RESULTS
# ==========================================

def save_results(history):
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

        ax.set_title(f"Advanced Fine-Tuning - {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            REPORT_DIR / f"finetuning_advanced_{metric}.png",
            dpi=150
        )
        plt.close(fig)

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(
        REPORT_DIR / "finetuning_advanced_history.json",
        "w"
    ) as file:
        json.dump(history_data, file, indent=2)


# ==========================================
# MAIN
# ==========================================

def main():
    print("\n=== TRAINING MODEL F ===")

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
            patience=6,
            mode="min",
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
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

    save_results(history)

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
        "model": "finetuning_advanced",
        "validation_accuracy": float(val_accuracy),
        "validation_loss": float(val_loss),
        "epochs_completed": len(history.history["loss"])
    }

    with open(
        REPORT_DIR / "finetuning_advanced_metrics.json",
        "w"
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\n=== HASIL MODEL F ===")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Model tersimpan: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
