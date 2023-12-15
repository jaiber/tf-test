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

    print("Loading data") # Hardcoded for now
    file = "experiments/Cuboid100Episodes/MasterJsonForEpisodes2023-12-08_22-24-37-645.json"
    data = DataLoader(file)
    (all_ids, all_encoding, all_row_pos, all_col_pos, all_obs) = data.load()


    config = GatoConfig.small()

    model = Transformer(config)
    model.compile(optimizer=AdamW(), loss="mean_absolute_error")

    x_train = np.random.random((1, 132, 768)).astype(np.float32)
    y_train = np.random.random((1, 132, 768)).astype(np.float32)
    print("x_train shape: {}".format(x_train.shape))
    print("y_train shape: {}".format(y_train.shape))

    print("Training model")
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    print("Running model in training mode")
    hidden_states = model(x_train)
    print("hidden_states shape: {}".format(hidden_states.shape))

    # Print model info
    # model.summary()

    print("Running model.evaluate")
    model.evaluate(x_train, y_train, verbose=2)

    print("Running model.predict")
    input = np.random.random((1, 132, 768)).astype(np.float32)
    y_pred = model.predict(input)
    print(y_pred.shape)

    # Softmax
    print("Running softmax")
    y = layers.Dense(1, activation="softmax")(y_pred)
    print(y.shape)
