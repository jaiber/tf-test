import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import tensorflow as tf

cce = losses.CategoricalCrossentropy(from_logits=True, axis=-1)
sce = losses.SparseCategoricalCrossentropy(from_logits=True)


def loss_function(y_true, y_pred):
    global cce
    global sce

    print("y_true shape: ", y_true.shape)
    #tf.print("y_true: ", y_true, summarize=2)

    print("y_pred shape: ", y_pred.shape)
    #tf.print("y_pred: ", y_pred, summarize=2)

    losses = sce(y_true, y_pred)
    #tf.print("losses shape: ", losses.shape)

    return losses

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
print("inputs shape: ", inputs.shape)
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(hidden_dim, num_heads)
print("x shape: ", x.shape)
sys.exit(0)
x = transformer_block(x)
outputs = Dense(vocab_size, activation='softmax')(x)
 
# define the model
model = Model(inputs=inputs, outputs=outputs)
 
# compile the model
#model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy')
model.compile(optimizer=Adam(), loss=loss_function)
 
# example training data
import numpy as np
x_train = np.random.randint(vocab_size, size=(1000, input_length))
y_train = np.random.randint(vocab_size, size=(1000, input_length, 1))
 
# train the model
model.fit(x_train, y_train, epochs=10, batch_size=1)