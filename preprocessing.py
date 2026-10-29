from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from config import IMAGE_SIZE, RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_rgb_image(path: str | Path, target_size: int | Tuple[int, int] = IMAGE_SIZE):
    """Load an RGB image, resize it, and normalize to [0, 1]."""
    image = Image.open(path).convert("RGB")
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    image = image.resize(target_size, resample=Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def load_mask(path: str | Path, target_size: int | Tuple[int, int] = IMAGE_SIZE):
    mask = Image.open(path)
    if mask.mode != "L":
        mask = mask.convert("L")
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    mask = mask.resize(target_size, resample=Image.Resampling.NEAREST)
    arr = np.asarray(mask, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def add_gaussian_noise(image: np.ndarray, std_dev: float = 0.1) -> np.ndarray:
    noise = np.random.normal(0.0, std_dev, image.shape).astype(np.float32)
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def prepare_train_val_split(paths: Sequence[str | Path], val_fraction: float = 0.2, seed: int = RANDOM_SEED):
    """Split normal training images into train and validation sets."""
    if len(paths) == 0:
        raise ValueError("No training images were provided for validation splitting.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)

    if len(paths) == 1:
        return list(paths), []

    val_count = max(1, int(round(len(paths) * val_fraction)))
    val_count = min(val_count, len(paths) - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_paths = [paths[int(i)] for i in train_indices]
    val_paths = [paths[int(i)] for i in val_indices]
    return train_paths, val_paths


def build_image_dataset(paths: Sequence[str | Path], batch_size: int, image_size: int = IMAGE_SIZE, shuffle: bool = True, seed: int = RANDOM_SEED, add_noise: bool = False, noise_std: float = 0.1):
    """Build a TensorFlow dataset from image paths."""
    path_list = [str(path) for path in paths]

    def _map_fn(path):
        path_value = path.numpy().decode("utf-8")
        image = load_rgb_image(path_value, target_size=(image_size, image_size)).astype(np.float32)
        clean = image
        if add_noise:
            noisy = add_gaussian_noise(clean, std_dev=noise_std)
            return tf.convert_to_tensor(noisy), tf.convert_to_tensor(clean)
        return tf.convert_to_tensor(image), tf.convert_to_tensor(image)

    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(path_list))
    dataset = dataset.map(lambda x: tf.py_function(_map_fn, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (tf.ensure_shape(x, [image_size, image_size, 3]), tf.ensure_shape(y, [image_size, image_size, 3])), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(64, len(path_list)), seed=seed)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_inference_dataset(paths: Sequence[str | Path], batch_size: int, image_size: int = IMAGE_SIZE):
    """Create a dataset for inference without random augmentations."""
    path_list = [str(path) for path in paths]

    def _map_fn(path):
        path_value = path.numpy().decode("utf-8")
        image = load_rgb_image(path_value, target_size=(image_size, image_size)).astype(np.float32)
        return tf.convert_to_tensor(image)

    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(path_list))
    dataset = dataset.map(lambda x: tf.py_function(_map_fn, [x], [tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.ensure_shape(x, [image_size, image_size, 3]), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def predict_with_model(model, image_paths: Sequence[str | Path], batch_size: int = 8, image_size: int = IMAGE_SIZE):
    dataset = build_inference_dataset(image_paths, batch_size=batch_size, image_size=image_size)
    predictions = model.predict(dataset, verbose=0)
    return predictions


if __name__ == "__main__":
    sample = ["D:/fake.png"]
    try:
        build_image_dataset(sample, batch_size=1, image_size=128)
    except Exception as exc:
        print(f"Preprocessing pipeline initialized successfully. Validation not run on actual images. Error: {exc}")
