#!/usr/bin/env python3

import sys
import argparse
import logging
import numpy as np
from gato import Gato
import tensorflow as tf
from config import GatoConfig
from loader import DataLoader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("Loading data")  # Hardcoded for now
    file = "experiments/Cuboid100Episodes/MasterJsonForEpisodes2023-12-08_22-24-37-645.json"
    data = DataLoader(file)
    (all_ids, all_encoding, all_row_pos, all_col_pos, all_obs) = data.load()

    config = GatoConfig.small()
    gato_model = Gato(config, trainable=True, name="Gato")

    all_ids = tf.expand_dims(all_ids, axis=0)
    all_encoding = tf.expand_dims(all_encoding, axis=0)
    all_row_pos = (
        tf.expand_dims(tf.cast(all_row_pos[0], tf.int32), axis=0),
        tf.expand_dims(tf.cast(all_row_pos[1], tf.int32), axis=0),
    )
    all_col_pos = (
        tf.expand_dims(tf.cast(all_col_pos[0], tf.int32), axis=0),
        tf.expand_dims(tf.cast(all_col_pos[1], tf.int32), axis=0),
    )
    all_obs = (
        tf.expand_dims(tf.cast(all_obs[0], tf.int32), axis=0),
        tf.expand_dims(tf.cast(all_obs[1], tf.int32), axis=0),
    )

    print("Running model in training mode")
    x_train = (all_ids, (all_encoding, all_row_pos, all_col_pos), all_obs)
    y_train = np.random.random((x_train[0].shape[0], 132, 768)).astype(np.float32)

    print("Compiling model ================")
    gato_model.compile(optimizer=tf.keras.optimizers.AdamW(), loss="mean_absolute_error")

    gato_model(x_train)
    print("model summary: ", gato_model.summary(expand_nested=False))

    print("Training the model ================")
    gato_model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    # sys.exit(0)

    print("Running model.evaluate =================")
    gato_model.evaluate(x_train, y_train, verbose=2)

    # print("Running model.predict")
    # input = np.random.random((1, 132, 768)).astype(np.float32)
    # y_pred = gato_model.predict(input)
    # print(y_pred.shape)

    # Softmax
    # print("Running softmax")
    # y = layers.Dense(1, activation="softmax")(y_pred)
    # print(y.shape)
