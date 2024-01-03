import tensorflow as tf

# Example data
y_true = tf.constant([[[1, 0, 0]]], dtype=tf.float32)  # One-hot encoded
y_pred = tf.constant([[[0, 1, 0]]], dtype=tf.float32)  # Class probabilities (logits)

# Compute categorical crossentropy loss
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)

print("Loss:", loss.numpy())
