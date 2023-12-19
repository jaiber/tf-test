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

    print("Running model in training mode")
    x_train = (all_ids, (all_encoding, all_row_pos, all_col_pos), all_obs)
    y_train = np.random.random((x_train[0].shape[0], 132, 768)).astype(np.float32)
    gato_model.compile(
        optimizer=tf.keras.optimizers.AdamW(), loss="mean_absolute_error"
    )

    gato_model(x_train)
    # print("model summary: ", gato_model.summary(expand_nested=True))
    # sys.exit(0)
    gato_model.fit(
        x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=2
    )

    sys.exit(0)

    embedding = gato_model.embedding(
        (all_ids, (all_encoding, all_row_pos, all_col_pos), all_obs)
    )
    logging.info("Embedding shape: {}".format(embedding.shape))
    x_train = embedding
    y_train = np.random.random(x_train.shape).astype(np.float32)

    print("Training model")
    gato_model.train_transformer(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        optimizer=tf.keras.optimizers.AdamW(),
        loss="mean_absolute_error",
    )

    # print("Running model in training mode")
    # hidden_states = gato_model(x_train)
    # print("hidden_states shape: {}".format(hidden_states.shape))

    # Print model info
    # model.summary()

    # print("Running model.evaluate")
    # gato_model.evaluate(x_train, y_train, verbose=2)

    # print("Running model.predict")
    # input = np.random.random((1, 132, 768)).astype(np.float32)
    # y_pred = gato_model.predict(input)
    # print(y_pred.shape)

    # Softmax
    # print("Running softmax")
    # y = layers.Dense(1, activation="softmax")(y_pred)
    # print(y.shape)
