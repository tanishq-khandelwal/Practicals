import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid activation function."""
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    """Initialize weights and biases."""
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    bias_hidden = np.ones((1, hidden_size))
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_output = np.ones((1, output_size))
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

def forward_propagation(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    """Perform forward propagation."""
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)
    return hidden_layer_output, output_layer_output

def calculate_loss(y_true, y_pred):
    """Calculate mean squared error loss."""
    return np.mean((y_true - y_pred) ** 2)

def update_weights(X, y, hidden_layer_output, output_layer_output, weights_hidden_output, weights_input_hidden, learning_rate):
    """Perform backpropagation to update weights and biases."""
    output_error = y - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
    
    return weights_input_hidden, weights_hidden_output

def train_neural_network(X, y, epochs, learning_rate):
    input_size = X.shape[1]
    hidden_size = 4  # Number of neurons in the hidden layer
    output_size = 1
    weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = initialize_weights(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        # Forward propagation
        hidden_layer_output, output_layer_output = forward_propagation(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
        
        # Calculate loss
        loss = calculate_loss(y, output_layer_output)
        
        # Backward propagation
        weights_input_hidden, weights_hidden_output = update_weights(X, y, hidden_layer_output, output_layer_output, weights_hidden_output, weights_input_hidden, learning_rate)
        
        # Print the loss every 1000 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

# Example usage
# Assuming X and y are your input features and target values, respectively
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
epochs = 1000
learning_rate = 0.1
trained_weights_input_hidden, trained_bias_hidden, trained_weights_hidden_output, trained_bias_output = train_neural_network(X, y, epochs, learning_rate)

# Test the trained neural network
hidden_layer_output, predictions = forward_propagation(X, trained_weights_input_hidden, trained_bias_hidden, trained_weights_hidden_output, trained_bias_output)
print("Predictions after training:")
print(predictions)
