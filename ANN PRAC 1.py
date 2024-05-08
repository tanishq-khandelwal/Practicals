import numpy as np
import matplotlib.pyplot as plt

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# def softmax(x):
#     exp_values = np.exp(x - np.max(x, axis=0))
#     return exp_values / np.sum(exp_values, axis=0)

# Generate data
x = np.linspace(-5, 5, 100)

# Compute activation functions
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
y_lin = linear(x)

# Plot the activation functions
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.plot(x, y_lin, label='Linear')
plt.title('Linear Activation Function')
plt.xlabel('x')
plt.ylabel('linear(x)')
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(x, y_relu, label='ReLU')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(x, y_tanh, label='Tanh')
plt.title('Tanh Activation Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.legend()

plt.tight_layout()
plt.show()