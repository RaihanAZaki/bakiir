
from pathlib import Path
import json
import shutil

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

# ============================================
# CONFIGURATION
# ============================================

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = ["Fresh", "Rotten"]

HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 15

HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_LAYERS = 30

TRAIN_DIR = Path("dataset/cleaned/train")
VALID_DIR = Path("dataset/cleaned/valid")

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/model_i")

HEAD_MODEL_PATH = MODEL_DIR / "binary_head_best.keras"
FINE_MODEL_PATH = MODEL_DIR / "binary_finetuned_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "binary_freshness.keras"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# ============================================
# LOAD DATASET
# ============================================

def get_files(folder):
    paths = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = folder / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(class_dir)

        files = sorted(
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in (
                ".jpg", ".jpeg", ".png"
            )
        )

        for path in files:
            paths.append(str(path))
            labels.append(label)

    if not paths:
        raise ValueError(f"Dataset kosong: {folder}")

    return paths, labels


def decode_image(path, label):
    content = tf.io.read_file(path)

    image = tf.io.decode_image(
        content,
        channels=3,
        expand_animations=False
    )

    image.set_shape((None, None, 3))

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear"
    )

    image = tf.cast(image, tf.float32)

    return image, label


def create_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, np.asarray(labels, dtype=np.int32))
    )

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)

    return ds.prefetch(tf.data.AUTOTUNE)


# ============================================
# CREATE CNN MODEL
# ============================================

def build_model():
    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="egg_image"
    )

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                0.5,
                fill_mode="constant",
                fill_value=0.0,
                seed=SEED
            ),
            tf.keras.layers.RandomZoom(
                0.08,
                fill_mode="constant",
                fill_value=0.0,
                seed=SEED + 1
            ),
            tf.keras.layers.RandomContrast(
                0.1,
                seed=SEED + 2
            )
        ],
        name="augmentation"
    )

    x = augmentation(inputs)

    # Input 0..255 menjadi -1..1.
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenet_preprocessing"
    )(x)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    backbone.trainable = False

    # Batch Normalization selalu inference mode
    # untuk menjaga statistik pretrained.
    x = backbone(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="freshness_prediction"
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name="Binary_Egg_Freshness"
    )


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def callbacks_for(path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(path),
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


def evaluate_loss(path, valid_ds):
    model = tf.keras.models.load_model(
        path,
        compile=False
    )

    compile_model(model, 1e-5)

    loss, accuracy = model.evaluate(
        valid_ds,
        verbose=0
    )

    return float(loss), float(accuracy)


# ============================================
# MAIN
# ============================================

def main():
    # Jangan menimpa model lama atau hasil
    # Model I yang mungkin sudah tersedia.
    for path in [
        HEAD_MODEL_PATH,
        FINE_MODEL_PATH,
        FINAL_MODEL_PATH
    ]:
        if path.exists():
            raise FileExistsError(
                f"File sudah ada: {path}. "
                "Training dibatalkan agar tidak menimpa model."
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
        valid_labels
    )

    print("\n=== DATASET MODEL I ===")
    print("Train:", len(train_paths))
    print("Validation:", len(valid_paths))
    print("Classes:", CLASS_NAMES)

    # ========================================
    # PHASE 1: TRANSFER LEARNING
    # ========================================

    print("\n=== PHASE 1: TRAIN CLASSIFIER ===")

    model = build_model()

    compile_model(model, HEAD_LR)

    history_head = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=HEAD_EPOCHS,
        callbacks=callbacks_for(HEAD_MODEL_PATH)
    )

    # ========================================
    # PHASE 2: FINE-TUNING
    # ========================================

    print("\n=== PHASE 2: FINE-TUNING ===")

    model = tf.keras.models.load_model(
        HEAD_MODEL_PATH,
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

    # Bekukan seluruh backbone terlebih dahulu.
    for layer in backbone.layers:
        layer.trainable = False

    # Buka 30 layer terakhir, kecuali BatchNorm.
    for layer in backbone.layers[-FINE_TUNE_LAYERS:]:
        if not isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = True

    compile_model(model, FINE_TUNE_LR)

    history_fine = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks_for(FINE_MODEL_PATH)
    )

    # ========================================
    # SELECT BEST CHECKPOINT
    # ========================================

    head_loss, head_acc = evaluate_loss(
        HEAD_MODEL_PATH,
        valid_ds
    )

    fine_loss, fine_acc = evaluate_loss(
        FINE_MODEL_PATH,
        valid_ds
    )

    if fine_loss < head_loss:
        selected_path = FINE_MODEL_PATH
        selected_stage = "fine_tuning"
    else:
        selected_path = HEAD_MODEL_PATH
        selected_stage = "transfer_learning"

    shutil.copy2(
        selected_path,
        FINAL_MODEL_PATH
    )

    # ========================================
    # FINAL VALIDATION METRICS
    # ========================================

    final_model = tf.keras.models.load_model(
        FINAL_MODEL_PATH,
        compile=False
    )

    probabilities = final_model.predict(
        valid_ds,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    actual = np.asarray(
        valid_labels,
        dtype=np.int32
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual,
            predictions,
            labels=[0, 1],
            average="macro",
            zero_division=0
        )
    )

    results = {
        "model": "binary_freshness",
        "classes": CLASS_NAMES,
        "training_images": len(train_paths),
        "validation_images": len(valid_paths),
        "selected_stage": selected_stage,
        "head_validation_loss": head_loss,
        "head_validation_accuracy": head_acc,
        "finetuning_validation_loss": fine_loss,
        "finetuning_validation_accuracy": fine_acc,
        "accuracy": float(
            accuracy_score(actual, predictions)
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(
            actual,
            predictions,
            labels=[0, 1]
        ).tolist(),
        "notes": (
            "Exploratory validation on Mendeley Fresh "
            "and Rotten labels. Not independently "
            "verified internal egg freshness."
        )
    }

    with open(
        REPORT_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    report = classification_report(
        actual,
        predictions,
        labels=[0, 1],
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
    ) as f:
        json.dump(history_data, f, indent=2)

    print("\n=== MODEL I RESULTS ===")
    print("Selected stage:", selected_stage)
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Macro F1: {results['macro_f1']:.2%}")
    print("Confusion matrix:")
    print(np.asarray(results["confusion_matrix"]))
    print("Saved:", FINAL_MODEL_PATH)


if __name__ == "__main__":
    main()
