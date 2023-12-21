#!/usr/bin/env python3

import sys
import argparse
import logging
import numpy as np
from gato import Gato
import tensorflow as tf
from config import GatoConfig
from loader import DataLoader

def loss_function(y_true, y_pred):

    print ("y_true shape: ", y_true.shape)
    print ("y_true: ", y_true)
    print ("y_pred shape: ", y_pred.shape)
    print ("y_pred: ", y_pred)

    loss = 0.01
    return loss

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
    (input_tokens, sequence_encoding, row_pos, col_pos, obs_encoding, all_discrete) = data.load()

    config = GatoConfig.small()
    gato_model = Gato(config, trainable=True, name="Gato")

    # Adding batch dimension
    input_tokens = tf.expand_dims(
        input_tokens, axis=0
    )  # Contains image, continuous, discrete tokens
    sequence_encoding = tf.expand_dims(sequence_encoding, axis=0)  # Contains sequence encoding
    row_pos = (  # Contains rows_from and rows_to
        tf.expand_dims(tf.cast(row_pos[0], tf.int32), axis=0),
        tf.expand_dims(tf.cast(row_pos[1], tf.int32), axis=0),
    )
    col_pos = (  # Contains cols_from and cols_to
        tf.expand_dims(tf.cast(col_pos[0], tf.int32), axis=0),
        tf.expand_dims(tf.cast(col_pos[1], tf.int32), axis=0),
    )
    obs_encoding = (  # Contains observation encoding and mask
        tf.expand_dims(tf.cast(obs_encoding[0], tf.int32), axis=0),
        tf.expand_dims(tf.cast(obs_encoding[1], tf.int32), axis=0),
    )

    x_train = (input_tokens, (sequence_encoding, row_pos, col_pos), obs_encoding)
    # y_train = np.random.random((1, input_tokens.shape[2], 768)).astype(np.float32)
    y_train = all_discrete
    print("all_discrete shape: ", all_discrete.shape)
    y_train = tf.transpose(all_discrete)
    print("all_discrete shape: ", all_discrete.shape)
    tf.print(y_train.numpy())

    print("Compiling model ================")
    gato_model.compile(optimizer=tf.keras.optimizers.AdamW(), loss="mean_absolute_error")

    gato_model(x_train)
    print("model summary: ", gato_model.summary(expand_nested=False))

    print("Training the model ================")
    gato_model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    print("Running model.evaluate =================")
    gato_model.evaluate(x_train, y_train, verbose=2)

    # print("Running model.predict")
    # input = np.random.random((1, 132, 768)).astype(np.float32)
    # y_pred = gato_model.predict(input)
    # print(y_pred.shape)
