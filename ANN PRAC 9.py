import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.rand(self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.random.rand(self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.predicted_output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        return self.predicted_output

    def backward(self, inputs, targets):
        # Calculate output layer error and delta
        output_error = targets - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        # Calculate hidden layer error and delta
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta) * self.learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta) * self.learning_rate

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            self.forward(inputs)
            self.backward(inputs, targets)

# Example usage:
if __name__ == "__main__":
    # Define input, hidden, and output layer sizes
    input_size = 2
    hidden_size = 4
    output_size = 1

    neural_net = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1)

    # Define input and target data
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Train the neural network
    neural_net.train(inputs, targets, epochs=1000)

    # Test the trained neural network
    for i in range(len(inputs)):
        print("Input:", inputs[i], "Target:", targets[i], "Predicted:", neural_net.forward(inputs[i]))
