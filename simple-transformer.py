#!/usr/bin/env python3

import sys
import argparse
import logging
import numpy as np
from gato import Gato
import tensorflow as tf
from config import GatoConfig
from loader import DataLoader
from tensorflow.keras import losses

cce = losses.CategoricalCrossentropy(from_logits=True, axis=-1)
sce = losses.SparseCategoricalCrossentropy(from_logits=True)


def loss_function(y_true, y_pred):
    global cce
    global sce

    #y_pred = tf.squeeze(y_pred, axis=0)
    #y_true = tf.squeeze(y_true, axis=0)

    #y_pred = tf.squeeze(y_pred, axis=0)
    #y_true = tf.squeeze(y_true, axis=0)

    #print("y_true shape: ", y_true.shape)
    tf.print("y_true: ", y_true)

    #print("y_pred shape: ", y_pred.shape)
    tf.print("y_pred: ", y_pred)

    #y_label = tf.argmax(y_true, axis=0)
    #print("y_label shape: ", y_label.shape)
    #tf.print("y_label: ", y_label)

    #losses = sce(y_true, y_pred)

    losses = cce(y_true, y_pred)
    #print("losses shape: ", losses.shape)
    tf.print("losses: ", losses)

    return losses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--sliding_batch_size", type=int, default=4)
    parser.add_argument(
        "--exp_config",
        type=str,
        default="experiments/Cuboid100Episodes/MasterJsonForEpisodes2023-12-08_22-24-37-645.json",
    )
    args = parser.parse_args()

    logging.info("Loading data")  # Hardcoded for now
    # file = "experiments/Cuboid100Episodes/MasterJsonForEpisodes2023-12-08_22-24-37-645.json"
    data = DataLoader(args.exp_config)

    (
        image_tokens,
        continuous_tokens,
        discrete_tokens,
        sequence_encoding,
        row_pos,
        col_pos,
        obs_encoding,
        action_tokens,
    ) = data.load(sliding_batch_size=args.sliding_batch_size, mask_discrete=False)

    config = GatoConfig.small()
    gato_model = Gato(config, trainable=True, name="Gato")

    # Convert to tensors of type int32
    row_pos = (  # Contains rows_from and rows_to
        tf.cast(row_pos[0], tf.int32),
        tf.cast(row_pos[1], tf.int32),
    )
    col_pos = (  # Contains cols_from and cols_to
        tf.cast(col_pos[0], tf.int32),
        tf.cast(col_pos[1], tf.int32),
    )
    obs_encoding = (  # Contains observation encoding and mask
        tf.cast(obs_encoding[0], tf.int32),
        tf.cast(obs_encoding[1], tf.int32),
    )

    x_train = [
        (image_tokens),  # continuous_tokens, discrete_tokens),
        (sequence_encoding, row_pos, col_pos),
        obs_encoding,
    ]

    print("image tokens shape: ", image_tokens.shape)

    x_train = [
        np.random.random((6, 20, 768)).astype(np.float32),
        (sequence_encoding, row_pos, col_pos),
        obs_encoding,
    ]
    y_train = np.random.randint(3, size=(image_tokens.shape[0]))
    print("y_train:", y_train)
    y_train = tf.one_hot(y_train, depth=3, dtype=tf.int32)
    print("y_train:", y_train.numpy())
    print("Y_train shape: ", y_train.shape)
    # Reshape (6, 3) to (6, 1, 3)
    y_train = tf.expand_dims(y_train, axis=1)

    #y_train = np.random.randint(3, size=(image_tokens.shape[0], 1, 3))

    #y_train = action_tokens
    logging.info("y_train shape: %s", y_train.shape)
    # tf.print(y_train.numpy())

    logging.info("Compiling model ================")
    gato_model.compile(
        optimizer=tf.keras.optimizers.AdamW(), loss=loss_function #, metrics=["accuracy"]
    )
    # gato_model.compile(
    #    optimizer=tf.keras.optimizers.AdamW(), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    # )

    # gato_model(x_train)

    logging.info("Training the model ================")
    gato_model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        shuffle=True,
    )
    print("model summary: ", gato_model.summary(expand_nested=False))
    # print("Running model.evaluate =================")
    # gato_model.evaluate(x_train, y_train, verbose=2)

    # print("Running model.predict")
    # input = np.random.random((1, 132, 768)).astype(np.float32)
    # y_pred = gato_model.predict(input)
    # print(y_pred.shape)
