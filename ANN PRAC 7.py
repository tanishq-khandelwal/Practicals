import numpy as np
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_derivative = lambda x: x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size, hidden_size, output_size = 2, 2, 1
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

epochs, learning_rate = 10000, 0.1
for _ in range(epochs):
    # Forward pass
    hidden_layer = sigmoid(np.dot(X, weights_input_hidden))
    predicted_output = sigmoid(np.dot(hidden_layer, weights_hidden_output))

    # Backpropagation
    output_error = y - predicted_output
    hidden_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer)

    # Update weights
    weights_hidden_output += hidden_layer.T.dot(output_error) * learning_rate
    weights_input_hidden += X.T.dot(hidden_error) * learning_rate

# Test the trained network
predicted_output = sigmoid(np.dot(sigmoid(np.dot(X, weights_input_hidden)), weights_hidden_output))

print("Input:")
print(X)
print("Predicted Output:")
print(predicted_output)
