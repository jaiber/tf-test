import tensorflow as tf

# Example tensor of shape (6, 1, 3)
input_tensor = tf.constant([[[1, 2, 3]],
                            [[4, 5, 6]],
                            [[7, 8, 9]],
                            [[10, 11, 12]],
                            [[13, 14, 15]],
                            [[16, 17, 18]]], dtype=tf.float32)

# Shift values up along axis=0
shifted_tensor = tf.concat([input_tensor[1:], tf.zeros_like(input_tensor[:1])], axis=0)

# Print the original and shifted tensors
print("Original Tensor:")
print(input_tensor.numpy())
print("\nShifted Tensor:")
print(shifted_tensor.numpy())
