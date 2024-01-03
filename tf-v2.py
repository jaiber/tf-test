#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers


class TransformerDecoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerDecoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.dec_layers = [
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, rate)
            for _ in range(num_layers)
        ]

    def call(self, inputs, training):
        x = inputs
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training)
        return x


def build_transformer_decoder(
    num_patches, num_layers, embed_dim, num_heads, ff_dim, num_classes, rate=0.1
):
    inputs = layers.Input(shape=(num_patches, embed_dim))
    x = TransformerDecoder(num_layers, embed_dim, num_heads, ff_dim, rate)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, axis=-1)
def loss_function(y_true, y_pred):
    global cce

    #print("y_true shape: ", y_true.shape)
    #tf.print("y_true: ", y_true, summarize=2)

    #print("y_pred shape: ", y_pred.shape)
    #tf.print("y_pred: ", y_pred, summarize=2)

    losses = cce(y_true, y_pred)
    #tf.print("losses shape: ", losses.shape)

    return losses

# Parameters for the transformer
num_patches = 64  # example value, depends on your patch extraction
embed_dim = 256  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 512  # Hidden layer size in feed forward network
num_layers = 4  # Number of transformer layers
num_classes = 10  # Number of robot action classes
dropout_rate = 0.1
# Build the model
model = build_transformer_decoder(
    num_patches, num_layers, embed_dim, num_heads, ff_dim, num_classes, dropout_rate
)
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.AdamW(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
# Model summary
model.summary()

# Create x_train and y_train
x_train = tf.random.uniform((100, 64, 256))
y_train = tf.random.uniform((100, 10), maxval=10, dtype=tf.int32)

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=16)

