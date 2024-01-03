import tensorflow as tf

# Define the shape of the empty tensor
empty_shape = (1, 1, 768)

# Create a list to store tensors
tensor_list = []

# Simulate adding tensors in a loop (replace this with your actual loop)
for i in range(5):
    # Generate a random tensor of shape (1, 1, 768)
    new_tensor = tf.random.normal(empty_shape)
    
    # Append the new_tensor to the list
    tensor_list.append(new_tensor)

# Concatenate tensors from the list along axis 1
result_tensor = tf.concat(tensor_list, axis=1)

# Print the shape of the result tensor
print("Result Tensor Shape:", result_tensor.shape)
