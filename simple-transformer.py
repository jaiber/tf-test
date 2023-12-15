#!/usr/bin/env python3

import sys
import numpy as np
import tensorflow as tf
from config import GatoConfig
from typing import Dict, Any, Union
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models, activations
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.optimizers import Adam, AdamW


# From Gato
class TransformerBlock(Model):
    def __init__(
        self,
        config: Union[GatoConfig, Dict[str, Any]],
        trainable: bool = True,
        name: str = None,
        **kwargs,
    ):
        super(TransformerBlock, self).__init__(trainable=trainable, name=name, **kwargs)
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.attention = self.feed_forward = self.dropout = None
        self.layer_norm1 = self.layer_norm2 = None

        self.dense = layers.Dense(hidden_dim)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        hidden_size = input_shape[-1]

        self.attention = layers.MultiHeadAttention(
            num_heads=self.config.num_attention_heads,
            key_dim=self.config.key_value_size,
            value_dim=self.config.key_value_size,
            dropout=self.config.dropout_rate,
            name="attention",
        )
        self.dropout = layers.Dropout(self.config.dropout_rate, name="attention_dropout")
        self.feed_forward = models.Sequential(
            layers=[
                layers.Dense(
                    units=self.config.feedforward_hidden_size,
                    activation="linear",
                    name="dense_intermediate",
                ),
                # Appendix C.1. Transformer Hyperparameters
                # Activation Function: GEGLU
                layers.Lambda(lambda x: activations.gelu(x, approximate=False), name="gelu"),
                layers.Dropout(self.config.dropout_rate, name="dropout_intermediate"),
                layers.Dense(units=hidden_size, activation="linear", name="dense"),
                layers.Dropout(self.config.dropout_rate, name="dropout"),
            ],
            name="feed_forward",
        )
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6, name="layer_norm1")
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6, name="layer_norm2")

    def call(self, inputs):
        # Appendix C.1. Transformer Hyperparameters
        # Layer Normalization: Pre-Norm
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.attention(x, x, x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(inputs)
        x = self.feed_forward(x)
        x = x + residual

        return x
        # attn_output = self.dense(inputs)
        # out = self.layernorm(tf.cast(inputs, float) + attn_output)
        # return out

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update(
            {
                "config": self.config.to_dict(),
            }
        )
        return config


config = GatoConfig.small()

vocab_size = 100  # TODO - change to 768 example vocabulary size
embed_dim = 128  # example embedding size
hidden_dim = 128  # example hidden dimension size
num_heads = 2  # example number of heads for attention
input_length = 128  # example input length


class Transformer(models.Model):
    def __init__(
        self,
        config: Union[GatoConfig, Dict[str, Any]],
        trainable: bool = True,
        name: str = None,
        **kwargs,
    ):
        super(Transformer, self).__init__(trainable=trainable, name=name, **kwargs)
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.encoders = [
            TransformerBlock(
                config=self.config, trainable=trainable, name="EncoderBlock{}".format(idx)
            )
            for idx in range(self.config.num_transformer_blocks)
        ]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for encoder in self.encoders:
            x = encoder(x)
        return x

    def get_config(self):
        return super(Transformer, self).get_config()


# Model definition
inputs = layers.Input(shape=(input_length,))

embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
x = embedding_layer(inputs)

transformer_block = TransformerBlock(config)
for encoder in range(config.num_transformer_blocks):
    x = transformer_block(x)

outputs = layers.Dense(vocab_size, activation="softmax")(x)

#------------------------------------------------------------
# define the model
# model = Model(inputs=inputs, outputs=outputs)

# compile the model
# model.compile(optimizer=AdamW(), loss="sparse_categorical_crossentropy")

# example training data
# x_train = np.random.randint(vocab_size, size=(100, input_length))
# y_train = np.random.randint(vocab_size, size=(100, input_length, 1))

# train the model
# model.fit(x_train, y_train, epochs=10, batch_size=32)
#------------------------------------------------------------

model = Transformer(config)
model.compile(optimizer=Adam(), loss="mean_absolute_error")

x_train = np.random.random((1, 132, 768)).astype(np.float32)
y_train = np.random.random((1, 132, 768)).astype(np.float32)

hidden_states = model(x_train)
print("hidden_states shape: {}".format(hidden_states.shape))

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("Training model")
model.fit(x_train, y_train, epochs=10, batch_size=32)

# model(input)
print("Done calling model with input")

# Print model info
# model.summary()

model.evaluate(x_train, y_train, verbose=2)

print("Running model.predict")
input = np.random.random((1, 132, 768)).astype(np.float32)
y_pred = model.predict(input)
print(y_pred.shape)

# Softmax
y = layers.Dense(1, activation="softmax")(y_pred)
print(y.shape)