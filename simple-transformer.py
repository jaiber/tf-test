#!/usr/bin/env python3

import sys
import numpy as np
import tensorflow as tf
from config import GatoConfig
from typing import Dict, Any, Union
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models, activations
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

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update(
            {
                "config": self.config.to_dict(),
            }
        )
        return config


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


if __name__ == "__main__":

    config = GatoConfig.small()

    model = Transformer(config)
    model.compile(optimizer=AdamW(), loss="mean_absolute_error")

    x_train = np.random.random((1, 132, 768)).astype(np.float32)
    y_train = np.random.random((1, 132, 768)).astype(np.float32)
    print("x_train shape: {}".format(x_train.shape))
    print("y_train shape: {}".format(y_train.shape))
  
    print("Training model")
    model.fit(x_train, y_train, epochs=10, batch_size=32)

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
