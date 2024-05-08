import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases for hidden layers
        self.weights = []
        self.biases = []
        input_layer_size = input_size
        for _ in range(hidden_layers):
            self.weights.append(np.random.randn(input_layer_size, hidden_size))
            self.biases.append(np.zeros((1, hidden_size)))
            input_layer_size = hidden_size
        
        # Initialize weights and biases for output layer
        self.weights.append(np.random.randn(hidden_size, output_size))
        self.biases.append(np.zeros((1, output_size)))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward_pass(self, X):
        self.layer_outputs = []
        input_data = X
        for i in range(self.hidden_layers):
            # Hidden layer calculations
            hidden_output = self.relu(np.dot(input_data, self.weights[i]) + self.biases[i])
            self.layer_outputs.append(hidden_output)
            input_data = hidden_output
        
        # Output layer calculations
        output = np.dot(input_data, self.weights[-1]) + self.biases[-1]
        self.layer_outputs.append(output)
        probabilities = self.sigmoid(output)
        return probabilities
    
    def backward_pass(self, X, y, learning_rate):
        num_examples = X.shape[0]
    
    # Calculate gradients for output layer
        output_error = self.layer_outputs[-1] - y
        output_delta = output_error / num_examples
    
    # Update weights and biases for output layer
        self.weights[-1] -= learning_rate * np.dot(self.layer_outputs[-2].T, output_delta)
        self.biases[-1] -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    
    # Calculate gradients for hidden layers
        hidden_errors = []
        hidden_errors.append(np.dot(output_delta, self.weights[-1].T))
    
        for i in range(self.hidden_layers - 1, 0, -1):
            delta = hidden_errors[-1] * (self.layer_outputs[i] > 0)
            hidden_errors.append(np.dot(delta, self.weights[i].T))
    
        hidden_errors.reverse()
    
    # Update weights and biases for hidden layers
        for i in range(self.hidden_layers):
            self.weights[i] -= learning_rate * np.dot(self.layer_outputs[i].T, hidden_errors[i])
            self.biases[i] -= learning_rate * np.sum(hidden_errors[i], axis=0, keepdims=True)

    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            probabilities = self.forward_pass(X)
            self.backward_pass(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = -np.sum(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)) / len(X)
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, X):
        probabilities = self.forward_pass(X)
        return np.round(probabilities)


# Taking user input for neural network architecture
input_size = int(input("Enter input size: "))
hidden_layers = int(input("Enter number of hidden layers: "))
hidden_size = int(input("Enter number of neurons in hidden layer: "))
output_size = int(input("Enter number of neurons in output layer: "))

# Assuming X_train, y_train, X_test, y_test are your training and testing data
# You can replace these with your actual data
X_train = np.random.randn(100, input_size)
y_train = np.random.randint(0, 2, (100, output_size))
X_test = np.random.randn(20, input_size)
y_test = np.random.randint(0, 2, (20, output_size))

learning_rate = 0.01
epochs = 1000

# Instantiate the neural network
model = NeuralNetwork(input_size, hidden_layers, hidden_size, output_size)

# Train the neural network
model.train(X_train, y_train, epochs, learning_rate)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy}')
