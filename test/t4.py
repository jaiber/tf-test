import tensorflow as tf

# Assuming your original tensor is named original_tensor
original_tensor = tf.constant(tf.random.normal((4, 5)))

# Duplicate each column twice
duplicated_columns = tf.tile(original_tensor[:, tf.newaxis, :], multiples=[1, 3, 1])

# Reshape to get the final tensor shape (4, 396)
new_tensor = tf.reshape(duplicated_columns, (4, -1))

# Print the shapes for verification
print("Original Tensor Shape:", original_tensor.shape)
print(original_tensor.numpy())
print("New Tensor Shape:", new_tensor.shape)
print(new_tensor.numpy())