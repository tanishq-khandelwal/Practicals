import numpy as np

# Define the perceptron class
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation >= 0 else 0

    def train(self, training_inputs, labels, epochs=10, learning_rate=0.1):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += learning_rate * (label - prediction) * inputs
                self.weights[0] += learning_rate * (label - prediction)

def decimal_to_binary_ascii(decimal):
    binary_representation = bin(decimal)[2:].zfill(8)  # Convert decimal to 8-bit binary ASCII
    return [int(bit) for bit in binary_representation]

# Training data in ASCII binary representation (assuming 8-bit ASCII)
training_data = np.array([
    decimal_to_binary_ascii(48),  # ASCII for '0'
    decimal_to_binary_ascii(49),  # ASCII for '1'
    # ... continue for '2' to '9'
])

# Labels: 1 for even, 0 for odd
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# Create and train the perceptron
input_size = len(training_data[0])
perceptron = Perceptron(input_size)
perceptron.train(training_data, labels)

# Test the trained perceptron with user input
user_input = int(input("Enter a number from 0 to 9: "))
if 0 <= user_input <= 9:
    test_number = decimal_to_binary_ascii(user_input)
    prediction = perceptron.predict(test_number)

    if prediction == 1:
        print("The number is even.")
    else:
        print("The number is odd.")
else:
    print("Invalid input. Please enter a number from 0 to 9.")