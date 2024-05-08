import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])
# Seed random numbers to make calculations deterministic
np.random.seed(1)
# Initialize weights randomly with mean 0
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# Weights from input to hidden layer
weights_input_hidden = 2 * np.random.random((input_neurons, hidden_neurons)) - 1

# Weights from hidden to output layer
weights_hidden_output = 2 * np.random.random((hidden_neurons, output_neurons)) - 1

# Training
epochs = 10000
learning_rate = 0.1
for epoch in range(epochs):
   # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    # Backpropagation
    # Calculate the error
    output_error = y - output_layer_output
    # Calculate adjustments for the output layer
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    # Backpropagate the error to the hidden layer
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Update weights
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

# Testing the trained network
test_input = np.array([[0,0],
                       [0,1],
                       [1,0],
                       [1,1]])

hidden_layer_input = np.dot(test_input, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
output_layer_output = sigmoid(output_layer_input)
print("Input:")
print(test_input)

print("Output after training:")
print(output_layer_output)