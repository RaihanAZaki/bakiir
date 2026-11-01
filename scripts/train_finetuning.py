
from pathlib import Path
import json

import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import load_dataset

# =====================================
# CONFIGURATION
# =====================================

SEED = 42
EPOCHS = 30
LEARNING_RATE = 1e-5
FINE_TUNE_LAYERS = 30

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

BASE_MODEL_PATH = MODEL_DIR / "rotation_focused.keras"
OUTPUT_PATH = MODEL_DIR / "finetuning.keras"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# =====================================
# ROTATION AUGMENTATION
# =====================================

rotation_layer = tf.keras.layers.RandomRotation(
    factor=0.5,
    fill_mode="constant",
    fill_value=0.0,
    seed=SEED
)


def augment_batch(images, labels):

    rotated = rotation_layer(
        images,
        training=True
    )

    # 50% gambar tetap asli,
    # 50% mengalami rotasi acak.
    mask = tf.random.uniform(
        shape=(tf.shape(images)[0], 1, 1, 1)
    ) < 0.5

    images = tf.where(
        mask,
        rotated,
        images
    )

    return images, labels


# =====================================
# DATASET
# =====================================

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


# =====================================
# FINE-TUNING MODEL
# =====================================

def build_finetuning_model():

    print("\nLoading Rotation Focused Model...")

    model = tf.keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False
    )

    # Cari backbone MobileNetV2
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
            "Periksa model.summary()."
        )

    backbone.trainable = True

    # Bekukan sebagian besar layer backbone.
    for layer in backbone.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    # BatchNormalization tetap frozen.
    # Ini penting saat fine-tuning pretrained CNN.
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

    trainable = sum(
        int(tf.size(weight))
        for weight in model.trainable_weights
    )

    print("\n=== FINE-TUNING CONFIG ===")
    print("Backbone:", backbone.name)
    print("Fine-tune layers:", FINE_TUNE_LAYERS)
    print("Learning rate:", LEARNING_RATE)
    print("Trainable parameters:", trainable)

    return model


# =====================================
# TRAINING GRAPHS
# =====================================

def save_graphs(history):

    for metric in ["accuracy", "loss"]:

        plt.figure(figsize=(9, 5))

        plt.plot(
            history.history[metric],
            label=f"Train {metric}"
        )

        plt.plot(
            history.history[f"val_{metric}"],
            label=f"Validation {metric}"
        )

        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"Fine-Tuning - {metric}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            REPORT_DIR / f"finetuning_{metric}.png",
            dpi=150
        )

        plt.close()


# =====================================
# MAIN TRAINING
# =====================================

def main():

    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(
            BASE_MODEL_PATH
        )

    print("\n=== TRAINING MODEL E ===")

    train_ds, valid_ds = create_datasets()

    model = build_finetuning_model()

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            str(OUTPUT_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min"
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
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

    save_graphs(history)

    with open(
        REPORT_DIR / "finetuning_history.json",
        "w"
    ) as file:
        json.dump(
            history.history,
            file,
            indent=2
        )

    # Evaluasi checkpoint terbaik pada validasi
    best_model = tf.keras.models.load_model(
        OUTPUT_PATH,
        compile=False
    )

    best_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    val_loss, val_accuracy = best_model.evaluate(
        valid_ds,
        verbose=1
    )

    results = {
        "model": "finetuning",
        "validation_accuracy": float(val_accuracy),
        "validation_loss": float(val_loss),
        "epochs_completed": len(
            history.history["loss"]
        )
    }

    with open(
        REPORT_DIR / "finetuning_metrics.json",
        "w"
    ) as file:
        json.dump(results, file, indent=2)

    print("\n=== HASIL MODEL E ===")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Model: {OUTPUT_PATH}")

    print("\nFine-tuning selesai!")


if __name__ == "__main__":
    main()
