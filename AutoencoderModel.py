import numpy as np

def set_params(in_dim, hid_dim):
    # encoder
    W1 = np.random.randn(in_dim, hid_dim) * np.sqrt(2. / in_dim)
    b1 = np.zeros((1, hid_dim))
    
    # decoder
    W2 = np.random.randn(hid_dim, in_dim) * np.sqrt(2. / hid_dim)
    b2 = np.zeros((1, in_dim))

    return W1, b1, W2, b2

def forward_pass(W1, b1, W2, b2, X):
    # encoder
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1) # use relu function

    # decoder
    Z2 = np.dot(A1, W2) + b2
    A2 = (1 / (1 + np.exp(-Z2))) # use sigmoid function

    return Z1, A1, Z2, A2

def back_pass(X_n, X_c, Z1, A1, A2, W2):
    m = X_c.shape[0] # get the size (for averages)

    # calc error at output
    dZ2 = A2 - X_c
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    # apply relu derivative
    relu_d = (Z1 > 0).astype(float)
    dZ1 = dA1 * relu_d
    dW1 = (1/m) * np.dot(X_n.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

def update_wts(W1, b1, W2, b2, dW1, db1, dW2, db2, a):
    W1 = W1 - a * dW1
    b1 = b1 - a * db1
    W2 = W2 - a * dW2
    b2 = b2 - a * db2

    return W1, b1, W2, b2

def mse(A2, X_c):
    return np.mean((A2 - X_c) ** 2)

