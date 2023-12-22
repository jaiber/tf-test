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

    print("[dice_loss] y_pred=",y_pred,"y_true=",y_true)
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=6)
    args = parser.parse_args()

    print("Loading data")  # Hardcoded for now
    file = "experiments/Cuboid100Episodes/MasterJsonForEpisodes2023-12-08_22-24-37-645.json"
    data = DataLoader(file)
    (input_tokens, sequence_encoding, row_pos, col_pos, obs_encoding, all_discrete) = data.load()

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

    """
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
    """

    x_train = [input_tokens, (sequence_encoding, row_pos, col_pos), obs_encoding]
    #y_train = np.random.random((1, 132, 3)).astype(np.float32)
    y_train = np.random.randint(3, size=(input_tokens.shape[0], 1, 3))
    print ("y_train shape: ", y_train.shape)
    
    #y_train = all_discrete
    #print("all_discrete shape: ", all_discrete.shape)
    #y_train = tf.transpose(y_train))
    #print("all_discrete shape: ", y_train.shape)
    #tf.print(y_train.numpy())
    #y_train = tf.one_hot(y_train, depth=3, dtype=tf.int32)
    #print("y_train shape: ", y_train.shape)
    #tf.print(y_train.numpy())

    #y_train = tf.reshape(y_train, (y_train.shape[1], y_train.shape[0], y_train.shape[2]))
    
    print("y_train shape: ", y_train.shape)
    #tf.print(y_train.numpy())

    # sys.exit(0)

    print("Compiling model ================")
    gato_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_function, metrics=['accuracy'])

    gato_model(x_train)
    print("model summary: ", gato_model.summary(expand_nested=False))

    print("Training the model ================")
    gato_model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    #print("Running model.evaluate =================")
    #gato_model.evaluate(x_train, y_train, verbose=2)

    # print("Running model.predict")
    # input = np.random.random((1, 132, 768)).astype(np.float32)
    # y_pred = gato_model.predict(input)
    # print(y_pred.shape)
