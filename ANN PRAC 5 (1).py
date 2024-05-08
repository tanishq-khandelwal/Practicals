import numpy as np

# Define the neural network architecture
input_size = 2
hidden_size = 3
output_size = 1

# Initialize weights and biases
weights_input_hidden = np.random.rand(input_size, hidden_size)
biases_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.rand(hidden_size, output_size)
biases_output = np.zeros((1, output_size))

# Define the activation function (e.g., sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the input data and corresponding target
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [1], [0]])

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_output
    predicted_output = sigmoid(output_layer_input)

    # Calculate loss (mean squared error)
    loss = np.mean(0.5 * (target - predicted_output) ** 2)

    # Backward propagation
    output_error = target - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)

    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    biases_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += input_data.T.dot(hidden_layer_delta) * learning_rate
    biases_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")

# Print the final predicted output
print("\nFinal Predictions:")
print(predicted_output)
