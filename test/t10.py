import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def transformer_decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="transformer_decoder"):
    inputs = keras.Input(shape=(None,), name="inputs")
    dec_padding_mask = keras.layers.Lambda(lambda x: tf.cast(tf.math.equal(x, 0), tf.float32))(inputs)

    embeddings = layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = layers.Dropout(dropout)(embeddings)

    for i in range(num_layers):
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        attention_output = attention(
            query=embeddings, value=embeddings, attention_mask=None, training=False
        )
        attention_output = layers.Dense(units, activation="relu")(attention_output)
        attention_output = layers.Dropout(dropout)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + embeddings)

        outputs = layers.Dense(units, activation="relu")(attention_output)
        outputs = layers.Dense(vocab_size)(outputs)
        outputs = layers.Dropout(dropout)(outputs)
        outputs = layers.LayerNormalization(epsilon=1e-6)(outputs + attention_output)

    # Define the model
    model = keras.Model(inputs=[inputs], outputs=[outputs], name=name)

    return model



# Define the hyperparameters
vocab_size = 10000
num_layers = 4
units = 512
d_model = 128
num_heads = 8
dropout = 0.1
batch_size = 64
epochs = 100

# Prepare the data
# Replace this with your own data
data = tf.random.uniform((1000, 10), maxval=vocab_size, dtype=tf.int64)
labels = tf.random.uniform((1000, 10), maxval=vocab_size, dtype=tf.int64)

# Split the data into training and validation sets
train_data = data[:800]
train_labels = labels[:800]
val_data = data[800:]
val_labels = labels[800:]

# Create the model
model = transformer_decoder(vocab_size, num_layers, units, d_model, num_heads, dropout)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_data,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_data, val_labels),
)

# Generate text sentences
# Replace this with your own data
input_sentence = tf.random.uniform((1, 10), maxval=vocab_size, dtype=tf.int64)
output_sentence = model.predict(input_sentence)
