import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam


class TransformerBlock(Model):
    def __init__(self, hidden_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dense = Dense(hidden_dim)
        self.layernorm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.dense(inputs)
        out = self.layernorm(inputs + attn_output)
        return out


vocab_size = 10000  # example vocabulary size
embed_dim = 128  # example embedding size
hidden_dim = 128  # example hidden dimension size
num_heads = 2  # example number of heads for attention
input_length = 100  # example input length

inputs = Input(shape=(input_length,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(hidden_dim, num_heads)
x = transformer_block(x)
outputs = Dense(vocab_size, activation="softmax")(x)

# define the model
model = Model(inputs=inputs, outputs=outputs)

# compile the model
model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy")

# example training data
x_train = np.random.randint(vocab_size, size=(1000, input_length))
y_train = np.random.randint(vocab_size, size=(1000, input_length, 1))

# train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Example inference
x_test = np.random.randint(vocab_size, size=(1, input_length))
y_pred = model.predict(x_test)
print(y_pred.shape)

y = Dense(1, activation="sigmoid")(y_pred)
print(y.shape)

