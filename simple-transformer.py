#!/usr/bin/env python3

import sys
import argparse
import logging
import numpy as np
from gato import Gato, Transformer
import tensorflow as tf
from config import GatoConfig
from loader import DataLoader
from typing import Dict, Any, Union
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models, activations
from tensorflow.keras.optimizers import Adam, AdamW


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
    gato_model = Gato(config)

    embedding = gato_model.embedding(
        (all_ids, (all_encoding, all_row_pos, all_col_pos), all_obs)
    )
    logging.info("Embedding shape: {}".format(embedding.shape))
    x_train = embedding
    y_train = np.random.random((1, 132, 768)).astype(np.float32)

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
