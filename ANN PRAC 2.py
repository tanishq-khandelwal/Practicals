import numpy as np

def mp_neuron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    output = 1 if weighted_sum >= threshold else 0
    return output

def and_not_gate():
    weights = [1, -1]
    threshold = 1

    try:
        x1 = float(input("Enter the value for x1 (0 or 1): "))
        x2 = float(input("Enter the value for x2 (0 or 1): "))

        if x1 not in [0, 1] or x2 not in [0, 1]:
            raise ValueError("Input values must be 0 or 1.")

        inputs = np.array([x1, x2])
        output = mp_neuron(inputs, weights, threshold)
        print(f"Output for {x1} ANDNOT {x2} = {output}")

        # Test all four conditions
        conditions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for condition in conditions:
            inputs = np.array(condition)
            result = mp_neuron(inputs, weights, threshold)
            print(f"Output for {condition[0]} ANDNOT {condition[1]} = {result}")

    except ValueError as e:
        print(f"Error: {e}. Please enter valid input values (0 or 1). Try again.")

# Test the ANDNOT gate with user input
and_not_gate()