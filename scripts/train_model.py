
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocessing import (
    load_dataset,
    data_augmentation,
    CLASS_NAMES,
)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)


def build_model(use_augmentation=False):
    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = inputs

    if use_augmentation:
        x = data_augmentation(x)

    # MobileNetV2 membutuhkan piksel -1 sampai 1
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_graph(history, name):
    plt.figure(figsize=(9, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy - {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / f"{name}_accuracy.png"
    )
    plt.close()

    plt.figure(figsize=(9, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss - {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / f"{name}_loss.png"
    )
    plt.close()


def train(name, use_augmentation, epochs):
    print(f"\n=== TRAINING {name.upper()} ===")

    train_ds = load_dataset(
        "train",
        shuffle=True
    )

    valid_ds = load_dataset("valid")

    model = build_model(use_augmentation)

    model.summary()

    model_path = MODEL_DIR / f"{name}.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    save_graph(history, name)

    with open(
        REPORT_DIR / f"{name}_history.json",
        "w"
    ) as file:
        json.dump(history.history, file, indent=2)

    # Evaluasi validasi saja selama pengembangan.
    # Test set disimpan untuk evaluasi akhir.
    best_model = tf.keras.models.load_model(
        model_path
    )

    val_loss, val_accuracy = best_model.evaluate(
        valid_ds,
        verbose=1
    )

    print("\n=== HASIL VALIDASI ===")
    print(f"Model: {name}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Model tersimpan: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--model",
    choices=[
        "baseline",
        "augmentation",
        "augmentation_light"
    ],
    required=True
)

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    args = parser.parse_args()

    train(
    name=args.model,
    use_augmentation=args.model != "baseline",
    epochs=args.epochs
)
