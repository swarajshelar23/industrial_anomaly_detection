from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from config import BETA, IMAGE_SIZE, LEARNING_RATE, RANDOM_SEED


def encoder_block(x, filters: int, kernel_size: int = 3):
    x = layers.Conv2D(filters, kernel_size, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, kernel_size, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


def decoder_block(x, filters: int, kernel_size: int = 3):
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(filters, kernel_size, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, kernel_size, activation="relu", padding="same")(x)
    return x


def build_ae_cnn(input_shape: Tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3), latent_dim: int = 256):
    inputs = layers.Input(shape=input_shape)
    x = encoder_block(inputs, 32)
    x = encoder_block(x, 64)
    x = encoder_block(x, 128)
    x = encoder_block(x, 256)

    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim, activation="relu")(x)
    x = layers.Dense(8 * 8 * 256, activation="relu")(x)
    x = layers.Reshape((8, 8, 256))(x)

    x = decoder_block(x, 256)
    x = decoder_block(x, 128)
    x = decoder_block(x, 64)
    x = decoder_block(x, 32)
    outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)

    model = Model(inputs, outputs, name="ae_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


def build_denoising_ae_cnn(input_shape: Tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3), latent_dim: int = 256):
    return build_ae_cnn(input_shape=input_shape, latent_dim=latent_dim)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@tf.keras.utils.register_keras_serializable(package="industrial_anomaly_detection")
class ConvVAE(tf.keras.Model):
    def __init__(self, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), latent_dim=128, beta=BETA, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, 3, activation="relu", padding="same"),
            layers.Conv2D(256, 3, activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
        ])

        self.mu = layers.Dense(latent_dim, name="z_mean")
        self.log_var = layers.Dense(latent_dim, name="z_log_var")

        self.decoder_input = layers.Dense(8 * 8 * 256, activation="relu")
        self.decoder = tf.keras.Sequential([
            layers.Reshape((8, 8, 256)),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(256, 3, activation="relu", padding="same"),
            layers.Conv2D(256, 3, activation="relu", padding="same"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.Conv2D(3, 3, activation="sigmoid", padding="same"),
        ])

    def encode(self, x):
        x = self.encoder(x)
        z_mean = self.mu(x)
        z_log_var = self.log_var(x)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def decode(self, z):
        z = self.decoder_input(z)
        return self.decoder(z)

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decode(z)
        return reconstruction

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(x)
            z = self.reparameterize(z_mean, z_log_var)
            reconstruction = self.decode(z)
            reconstruction_loss = tf.reduce_mean(tf.square(y - reconstruction), axis=(1, 2, 3))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            total_loss = tf.reduce_mean(reconstruction_loss + self.beta * kl_loss)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "reconstruction_loss": tf.reduce_mean(reconstruction_loss), "kl_loss": tf.reduce_mean(kl_loss)}


def build_vae_cnn(input_shape: Tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3), latent_dim: int = 128, beta: float = BETA):
    model = ConvVAE(input_shape=input_shape, latent_dim=latent_dim, beta=beta, name="vae_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


def load_model_keras(model_path: str):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"ConvVAE": ConvVAE},
        compile=False,
    )


def model_name_to_builder(model_name: str):
    if model_name == "ae_cnn":
        return build_ae_cnn
    if model_name == "denoising_ae_cnn":
        return build_denoising_ae_cnn
    if model_name == "vae_cnn":
        return build_vae_cnn
    raise ValueError(f"Unsupported model requested: {model_name}")


if __name__ == "__main__":
    model = build_ae_cnn()
    print(model.summary())
