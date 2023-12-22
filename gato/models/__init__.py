import sys
import tensorflow as tf

from gato.models.transformer import TransformerBlock
from gato.models.embedding import (
    PatchPositionEncoding,
    ResidualEmbedding,
    LocalPositionEncoding,
    DiscreteEmbedding,
)
from gato.models.tokenizers import ContinuousValueTokenizer

from tensorflow.keras import models, layers
from gato import GatoConfig
from typing import Dict, Any, Union


class Gato(models.Model):
    def __init__(
        self,
        config: Union[GatoConfig, Dict[str, Any]],
        trainable: bool = True,
        name: str = "Gato",
        **kwargs
    ):
        super(Gato, self).__init__(trainable=trainable, name=name, **kwargs)
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.image_embedding = PatchEmbedding(
            config, trainable=trainable, name="ImagePatchEmbedding"
        )
        self.discrete_embedding = DiscreteEmbedding(
            config, trainable=trainable, name="DiscreteEmbedding"
        )
        self.continuous_encoding = ContinuousValueTokenizer(config, name="ContinuousValueEncoding")
        self.transformer = Transformer(config, trainable=trainable, name="Transformers")
        self.local_pos_encoding = LocalPositionEncoding(
            config, trainable=trainable, name="LocalPositionEncoding"
        )

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(3, activation="softmax", name="Output")

    def call(self, inputs, training=None, mask=None):
        # input_ids with (B, L, 768)
        # encoding with (B, L) or (B,)
        # row_pos and col_pos with tuple of (pos_from, pos_to)
        # obs_pos and obs_mask with (B, L) or (B,)
        input_ids, (encoding, row_pos, col_pos), (obs_pos, obs_mask) = inputs

        """
        # Strip batch dinmension
        
        input_ids = tf.squeeze(input_ids, axis=0)
        encoding = tf.squeeze(encoding, axis=0)
        row_pos = (
            tf.cast(tf.squeeze(row_pos[0], axis=0), tf.int32),
            tf.cast(tf.squeeze(row_pos[1], axis=0), tf.int32),
        )
        col_pos = (
            tf.cast(tf.squeeze(col_pos[0], axis=0), tf.int32),
            tf.cast(tf.squeeze(col_pos[1], axis=0), tf.int32),
        )
        obs_pos = tf.cast(tf.squeeze(obs_pos, axis=0), tf.int32)
        obs_mask = tf.cast(tf.squeeze(obs_mask, axis=0), tf.int32)
        """

        print("    input_ids shape: ", input_ids.shape)
        print("    encoding shape: ", encoding.shape)
        print("    row_pos shape: ", row_pos[0].shape)
        # print ("    col_pos shape: ", col_pos[0].shape)
        # print ("    obs_pos shape: ", obs_pos.shape)
        # print ("    obs_mask shape: ", obs_mask.shape)
        # Encoding flags for embed masks
        # 0 - image
        # 1 - continuous
        # 2 - discrete (actions, texts)
        encoding = tf.one_hot(encoding, depth=3, dtype=tf.float32)

        ones = tf.ones(
            (input_ids.shape[0], input_ids.shape[0], self.config.layer_width), dtype=tf.float32
        )
        image_embed = self.image_embedding((input_ids, (row_pos, col_pos)), training=training)
        print("multiplying: ", encoding[..., 0].transpose().shape, ones.shape)
        image_embed *= encoding[..., 0].transpose().matmul(ones)  # image patch masking

        # continuous value takes from first value of input_ids
        continuous_embed = self.continuous_encoding(input_ids[..., 0])
        continuous_embed = self.discrete_embedding(continuous_embed)
        continuous_embed *= encoding[..., 1].transpose().matmul(ones)  # continuous value masking

        discrete_embed = self.discrete_embedding(input_ids[..., 0])
        discrete_embed *= encoding[..., 2].transpose().matmul(ones)  # discrete value masking

        # Appendix C.3. Position Encodings > Local Observation Position Encodings
        # add local observation position encodings
        embed = image_embed + continuous_embed + discrete_embed
        embed += self.local_pos_encoding((obs_pos, obs_mask))
        hidden_states = self.transformer(embed)

        output = self.flatten(hidden_states)
        # Add dense softmax layer
        output = self.dense(output)
        # argmax
        # output = tf.argmax(output, axis=-1)
        print("output shape: ", output.shape)
        tf.print(output)
        return output

    def train_transformer(
        self,
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=2,
        optimizer=tf.keras.optimizers.AdamW(),
        loss="mean_absolute_error",
    ):
        self.transformer.compile(optimizer=optimizer, loss=loss)
        self.transformer.fit(
            x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose
        )

    def get_config(self):
        return super(Gato, self).get_config()


class Transformer(models.Model):
    def __init__(
        self,
        config: Union[GatoConfig, Dict[str, Any]],
        trainable: bool = True,
        name: str = None,
        **kwargs
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


class PatchEmbedding(models.Model):
    def __init__(
        self,
        config: Union[GatoConfig, Dict[str, Any]],
        trainable: bool = True,
        name: str = None,
        **kwargs
    ):
        super(PatchEmbedding, self).__init__(trainable=trainable, name=name, **kwargs)
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.residual_embedding = ResidualEmbedding(
            config, trainable=trainable, name="ResidualEmbedding"
        )
        self.pos_encoding = PatchPositionEncoding(
            config, trainable=trainable, name="PatchPositionEncoding"
        )

    def call(self, inputs, training=None, mask=None):
        input_ids, (row_pos, col_pos) = inputs
        patch_size = self.config.img_patch_size
        depth = self.config.input_dim // (patch_size * patch_size)

        x = input_ids.reshape((-1, input_ids.shape[1], patch_size, patch_size, depth))
        x = self.residual_embedding(x)
        x = self.pos_encoding((x, (row_pos, col_pos)))
        return x

    def get_config(self):
        return super(PatchEmbedding, self).get_config()
