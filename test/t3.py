import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class ImageTransformerBlock(layers.Layer):
    def __init__(self, num_heads=8, ff_dim=1024, dropout_rate=0.1):
        super(ImageTransformerBlock, self).__init__()

        self.conv_layer = layers.Conv2D(256, (3, 3), activation="relu")
        self.batch_norm = layers.BatchNormalization()
        self.max_pooling = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(ff_dim, activation="relu")

    def call(self, inputs):
        x = self.conv_layer(inputs)
        x = self.batch_norm(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class ActionTransformerBlock(layers.Layer):
    def __init__(self, ff_dim=64, dropout_rate=0.1):
        super(ActionTransformerBlock, self).__init__()

        self.dense = layers.Dense(ff_dim, activation="relu")

    def call(self, inputs):
        x = self.dense(inputs)
        return x

class MultimodalTransformerModel(tf.keras.Model):
    def __init__(self, input_shape_images, input_shape_actions, output_shape_rotations):
        super(MultimodalTransformerModel, self).__init__()

        self.image_transformer = ImageTransformerBlock()
        self.action_transformer = ActionTransformerBlock()
        self.concat = layers.Concatenate(axis=-1)
        self.transformer_layer = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)
        self.dropout = layers.Dropout(0.1)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.fc_rotation = layers.Dense(output_shape_rotations, activation="linear")

    def call(self, inputs):
        input_images, input_actions = inputs

        # Image transformer block
        x_images = self.image_transformer(input_images)

        # Action transformer block
        x_actions = self.action_transformer(input_actions)

        # Concatenate image and action features
        x = self.concat([x_images, x_actions])

        # Transformer layer
        x = self.transformer_layer(x, x)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # Fully connected layer for rotation prediction
        x = self.fc_rotation(x)

        return x

# Generate random sample data
num_samples = 100
image_shape = (64, 64, 3)
action_shape = (10,)
rotation_shape = 1

X_images = np.random.rand(num_samples, *image_shape)
X_actions = np.random.rand(num_samples, *action_shape)
y_rotations = np.random.rand(num_samples, rotation_shape)

# Normalize the data (optional)
X_images = X_images / 255.0
X_actions = (X_actions - np.mean(X_actions)) / np.std(X_actions)
y_rotations = (y_rotations - np.mean(y_rotations)) / np.std(y_rotations)

# Create the model
input_shape_images = image_shape
input_shape_actions = action_shape
output_shape_rotations = rotation_shape

model = MultimodalTransformerModel(input_shape_images, input_shape_actions, output_shape_rotations)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model with the sample data
model.fit([X_images, X_actions], y_rotations, epochs=10, batch_size=32)