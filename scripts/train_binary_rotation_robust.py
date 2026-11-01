
from pathlib import Path
import json
import shutil

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["Fresh", "Rotten"]

HEAD_EPOCHS = 15
FINE_EPOCHS = 20

HEAD_LR = 1e-4
FINE_LR = 5e-6
FINE_TUNE_LAYERS = 30

TRAIN_DIR = Path("dataset/cleaned/train")
VALID_DIR = Path("dataset/cleaned/valid")

BASE_MODEL = Path("models/rotation_robust.keras")

HEAD_MODEL = Path(
    "models/binary_rotation_head.keras"
)
FINE_MODEL = Path(
    "models/binary_rotation_finetuned.keras"
)
FINAL_MODEL = Path(
    "models/binary_rotation_robust.keras"
)

REPORT_DIR = Path("reports/model_j")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)


# ==========================================
# DATASET
# ==========================================

def get_files(directory):
    paths = []
    labels = []

    for label_id, class_name in enumerate(CLASS_NAMES):
        folder = directory / class_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        files = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in (
                ".jpg", ".jpeg", ".png"
            )
        )

        paths.extend(str(p) for p in files)
        labels.extend([label_id] * len(files))

    if not paths:
        raise ValueError(f"Dataset kosong: {directory}")

    return paths, np.asarray(labels, dtype=np.int32)


def decode_image(path, label):
    content = tf.io.read_file(path)

    image = tf.io.decode_image(
        content,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear"
    )

    return tf.cast(image, tf.float32), label


def augment_image(image, label):
    # 40% gambar tetap asli.
    should_augment = tf.random.uniform([]) < 0.60

    def transform():
        result = image

        # Rotasi 0, 90, 180 atau 270 derajat.
        k = tf.random.uniform(
            [],
            minval=0,
            maxval=4,
            dtype=tf.int32
        )

        result = tf.image.rot90(result, k=k)

        result = tf.image.random_brightness(
            result,
            max_delta=10.0
        )

        result = tf.image.random_contrast(
            result,
            lower=0.92,
            upper=1.08
        )

        return tf.clip_by_value(
            result, 0.0, 255.0
        )

    image = tf.cond(
        should_augment,
        transform,
        lambda: image
    )

    return image, label


def create_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        ds = ds.shuffle(
            len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        ds = ds.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(BATCH_SIZE)

    return ds.prefetch(tf.data.AUTOTUNE)


# ==========================================
# CREATE BINARY MODEL FROM MODEL G
# ==========================================

def build_binary_model():
    if not BASE_MODEL.is_file():
        raise FileNotFoundError(BASE_MODEL)

    original = tf.keras.models.load_model(
        BASE_MODEL,
        compile=False
    )

    # Temukan layer pooling terakhir Model G.
    pooling_layers = [
        layer
        for layer in original.layers
        if isinstance(
            layer,
            tf.keras.layers.GlobalAveragePooling2D
        )
    ]

    if not pooling_layers:
        print("Model G layers:")
        for layer in original.layers:
            print(layer.name, type(layer).__name__)

        raise ValueError(
            "GlobalAveragePooling2D tidak ditemukan. "
            "Struktur Model G perlu diperiksa."
        )

    pooling = pooling_layers[-1]

    # Gunakan seluruh graph Model G sampai pooling,
    # termasuk preprocessing dan backbone.
    features = pooling.output

    x = tf.keras.layers.Dropout(
        0.3,
        name="binary_dropout"
    )(features)

    output = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="binary_prediction"
    )(x)

    model = tf.keras.Model(
        inputs=original.inputs,
        outputs=output,
        name="Binary_Rotation_Robust"
    )

    # Bekukan feature extractor pada fase pertama.
    for layer in model.layers:
        layer.trainable = False

    model.get_layer("binary_prediction").trainable = True

    return model


def get_backbone(model):
    for layer in model.layers:
        if (
            isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        ):
            return layer

    raise ValueError(
        "Backbone MobileNetV2 tidak ditemukan."
    )


def enable_fine_tuning(model):
    # Bekukan layer utama.
    for layer in model.layers:
        layer.trainable = False

    backbone = get_backbone(model)
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    # Buka 30 layer terakhir kecuali BN.
    for layer in backbone.layers[-FINE_TUNE_LAYERS:]:
        if not isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = True

    model.get_layer("binary_prediction").trainable = True

    return model


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def make_callbacks(path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


def evaluate_checkpoint(path, dataset):
    model = tf.keras.models.load_model(
        path,
        compile=False
    )

    compile_model(model, 1e-5)

    loss, accuracy = model.evaluate(
        dataset,
        verbose=0
    )

    return float(loss), float(accuracy)


# ==========================================
# MAIN
# ==========================================

def main():
    # Mencegah hasil sebelumnya tertimpa.
    for path in [HEAD_MODEL, FINE_MODEL, FINAL_MODEL]:
        if path.exists():
            raise FileExistsError(
                f"File sudah ada: {path}"
            )

    train_paths, train_labels = get_files(TRAIN_DIR)
    valid_paths, valid_labels = get_files(VALID_DIR)

    train_ds = create_dataset(
        train_paths,
        train_labels,
        training=True
    )

    valid_ds = create_dataset(
        valid_paths,
        valid_labels,
        training=False
    )

    print("\n=== MODEL J DATASET ===")
    print("Train:", len(train_paths))
    print("Validation:", len(valid_paths))
    print("Classes:", CLASS_NAMES)

    # ======================================
    # PHASE 1 - HEAD TRAINING
    # ======================================

    print("\n=== PHASE 1: HEAD TRAINING ===")

    model = build_binary_model()
    compile_model(model, HEAD_LR)

    history_head = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=HEAD_EPOCHS,
        callbacks=make_callbacks(HEAD_MODEL)
    )

    # ======================================
    # PHASE 2 - FINE TUNING
    # ======================================

    print("\n=== PHASE 2: FINE TUNING ===")

    model = tf.keras.models.load_model(
        HEAD_MODEL,
        compile=False
    )

    model = enable_fine_tuning(model)
    compile_model(model, FINE_LR)

    history_fine = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=FINE_EPOCHS,
        callbacks=make_callbacks(FINE_MODEL)
    )

    # ======================================
    # SELECT CHECKPOINT
    # ======================================

    head_loss, head_acc = evaluate_checkpoint(
        HEAD_MODEL, valid_ds
    )

    fine_loss, fine_acc = evaluate_checkpoint(
        FINE_MODEL, valid_ds
    )

    if fine_loss < head_loss:
        chosen = FINE_MODEL
        stage = "fine_tuning"
    else:
        chosen = HEAD_MODEL
        stage = "head_training"

    shutil.copy2(chosen, FINAL_MODEL)

    final_model = tf.keras.models.load_model(
        FINAL_MODEL,
        compile=False
    )

    probabilities = final_model.predict(
        valid_ds,
        verbose=0
    )

    predictions = probabilities.argmax(axis=1)

    actual = valid_labels

    metrics = {
        "model": "binary_rotation_robust",
        "base_model": str(BASE_MODEL),
        "classes": CLASS_NAMES,
        "train_count": len(train_paths),
        "validation_count": len(valid_paths),
        "selected_stage": stage,
        "head_validation_loss": head_loss,
        "head_validation_accuracy": head_acc,
        "fine_validation_loss": fine_loss,
        "fine_validation_accuracy": fine_acc,
        "accuracy": float(
            accuracy_score(actual, predictions)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predictions,
            labels=[0, 1]
        ).tolist()
    }

    with open(
        REPORT_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        actual,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    (REPORT_DIR / "classification_report.txt").write_text(
        report,
        encoding="utf-8"
    )

    history_data = {
        "head": {
            key: [float(v) for v in values]
            for key, values in history_head.history.items()
        },
        "fine_tuning": {
            key: [float(v) for v in values]
            for key, values in history_fine.history.items()
        }
    }

    with open(
        REPORT_DIR / "training_history.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(history_data, file, indent=2)

    print("\n=== MODEL J RESULT ===")
    print("Selected stage:", stage)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro F1: {metrics['macro_f1']:.2%}")
    print("Confusion matrix:")
    print(np.asarray(metrics["confusion_matrix"]))
    print("Saved:", FINAL_MODEL)


if __name__ == "__main__":
    main()
