import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the input data
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Define the logistic regression model
logistic_model = Sequential([
    Dense(10, activation='softmax')
])

# Compile the logistic regression model
logistic_model.compile(optimizer=Adam(learning_rate=0.001),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

# Train the logistic regression model
logistic_model.fit(x_train, y_train, validation_split=0.1, epochs=10, batch_size=32)

# Evaluate the logistic regression on the test set
test_loss, test_acc = logistic_model.evaluate(x_test, y_test)
print("Logistic regression test loss:", test_loss)
print("Logistic regression test accuracy:", test_acc)
