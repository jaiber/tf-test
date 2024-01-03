#!/usr/bin/env python3

import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerModel(keras.Model):
    def __init__(
        self, embed_dim, num_heads, ff_dim, input_shape, num_classes, dropout_rate=0.1
    ):
        super(TransformerModel, self).__init__()

        self.embedding_layer = layers.Embedding(input_dim=10000, output_dim=embed_dim)
        self.dropout_embedding = layers.Dropout(dropout_rate)

        self.transformer_blocks = []
        for _ in range(num_heads):
            self.transformer_blocks.append(
                layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=embed_dim // num_heads
                )
            )
            self.transformer_blocks.append(layers.Dropout(dropout_rate))
            self.transformer_blocks.append(layers.LayerNormalization(epsilon=1e-6))

        self.ff_layer = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
        self.dropout_ff = layers.Dropout(dropout_rate)
        self.layer_norm_ff = layers.LayerNormalization(epsilon=1e-6)

        self.global_avg_pooling = layers.GlobalAveragePooling1D()

        self.dense1 = layers.Dense(20, activation="relu")
        self.dropout_dense1 = layers.Dropout(dropout_rate)

        self.output_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.dropout_embedding(x)

        for i in range(0, len(self.transformer_blocks), 3):
            # Use the correct arguments for MultiHeadAttention layer
            x = self.transformer_blocks[i](x, x, x)
            x = self.transformer_blocks[i + 1](x)
            x = self.transformer_blocks[i + 2](x)

        x = self.ff_layer(x)
        x = self.dropout_ff(x)
        x = self.layer_norm_ff(x)

        x = self.global_avg_pooling(x)

        x = self.dense1(x)
        x = self.dropout_dense1(x)

        return self.output_layer(x)


# Sample data loading and preprocessing
#imdb = keras.datasets.imdb
#(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

max_len = 200  # Maximum sequence length
#train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=max_len)
#test_data = keras.preprocessing.sequence.pad_sequences(test_data, maxlen=max_len)

# Trying with random data
train_data = tf.random.uniform((25000, 200), dtype=tf.float32)
test_data = tf.random.uniform((25000, 200), dtype=tf.float32)
train_labels = tf.random.uniform((25000,), minval=0, maxval=2, dtype=tf.int32)
test_labels = tf.random.uniform((25000,), minval=0, maxval=2, dtype=tf.int32)
print("train_data: ", train_data.shape)
print("train_labels: ", train_labels.shape)
print("test_data: ", test_data.shape)
print("test_labels: ", test_labels.shape)
#sys.exit(0)


# Model configuration
embed_dim = 32
num_heads = 2
ff_dim = 32
input_shape = (max_len,)
num_classes = 2

# Instantiate the model
model = TransformerModel(embed_dim, num_heads, ff_dim, input_shape, num_classes)

# Compile and summarize the model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# Training the model
print("Training the model")
model.fit(train_data, train_labels, epochs=3, batch_size=32, validation_split=0.2)
model.summary()

# Evaluate on test data
print("Evaluate on test data")
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc}")
