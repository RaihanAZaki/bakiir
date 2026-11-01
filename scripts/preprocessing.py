
from pathlib import Path
import tensorflow as tf

DATASET_DIR = Path("dataset/cleaned")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

CLASS_NAMES = [
    "Fresh",
    "Half-Fresh",
    "Rotten"
]

def load_dataset(split, shuffle=False):

    dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / split,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )

    return dataset.prefetch(
        tf.data.AUTOTUNE
    )


# Augmentation hanya untuk training
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(
        factor=0.5,
        fill_mode="reflect"
    ),
    tf.keras.layers.RandomFlip(
        mode="horizontal_and_vertical"
    ),
    tf.keras.layers.RandomZoom(
        height_factor=0.1,
        width_factor=0.1
    ),
    tf.keras.layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    ),
    tf.keras.layers.RandomContrast(
        factor=0.1
    )
], name="data_augmentation")


if __name__ == "__main__":

    train_ds = load_dataset(
        "train",
        shuffle=True
    )

    valid_ds = load_dataset("valid")
    test_ds = load_dataset("test")

    print("\n=== PREPROCESSING ===")
    print("Classes:", CLASS_NAMES)
    print("Image size:", IMG_SIZE)
    print("Batch size:", BATCH_SIZE)

    print("\nJumlah batch:")
    print("Train:", len(train_ds))
    print("Valid:", len(valid_ds))
    print("Test:", len(test_ds))

    for images, labels in train_ds.take(1):
        print("\nImage shape:", images.shape)
        print("Label shape:", labels.shape)

    print("\nPreprocessing berhasil!")
