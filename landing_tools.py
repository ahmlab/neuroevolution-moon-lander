import numpy as np

# Network Architecture
INPUT_SIZE = 8
HIDDEN_SIZE = 16
OUTPUT_SIZE = 2

# This function will unflatten the neural network from a 1D list
def unflatten(weights):
    # idx is a cursor that reads the values from the float list of the individual
    idx = 0
    W1 = weights[idx:idx+INPUT_SIZE*HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
    idx += INPUT_SIZE*HIDDEN_SIZE 
    b1 = weights[idx:idx+HIDDEN_SIZE]
    idx += HIDDEN_SIZE
    W2 = weights[idx:idx+HIDDEN_SIZE*OUTPUT_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    idx += HIDDEN_SIZE*OUTPUT_SIZE
    b2 = weights[idx:idx+OUTPUT_SIZE]
    return W1, b1, W2, b2

# Forward pass (inference): observation -> motor commands.
# No learning happens here. the GA in train.py evolves the weights.
def forward(obs, weights):
    W1, b1, W2, b2 = unflatten(np.array(weights))
    h = np.tanh(obs @ W1 + b1)
    out = np.tanh(h @ W2 + b2)
    return out