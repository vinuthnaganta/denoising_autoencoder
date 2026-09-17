import numpy as np

def leaky_relu(Z):
    return np.where(Z > 0, Z, Z * 0.01)

def leaky_relu_deriv(Z):
    return np.where(Z > 0, 1, 0.01)

def set_params(inp, h1, h2, h3):
    # # encoder
    # W1 = np.random.randn(in_dim, hid_dim) * np.sqrt(2. / in_dim)
    # b1 = np.zeros((1, hid_dim))
    
    # # decoder
    # W2 = np.random.randn(hid_dim, in_dim) * np.sqrt(2. / hid_dim)
    # b2 = np.zeros((1, in_dim))

    # return W1, b1, W2, b2

    # encoder with three hidden layers
    # he initial bc of relu func
    W1 = np.random.randn(inp, h1) * np.sqrt(2. / inp)
    b1 = np.zeros((1, h1))
    W2 = np.random.randn(h1, h2) * np.sqrt(2. / h1)
    b2 = np.zeros((1, h2))
    W3 = np.random.randn(h2, h3) * np.sqrt(2. / h2)
    b3 = np.zeros((1, h3))

    # decoder
    W4 = np.random.randn(h3, h2) * np.sqrt(2. / h3)
    b4 = np.zeros((1, h2))
    W5 = np.random.randn(h2, h1) * np.sqrt(2. / h2)
    b5 = np.zeros((1, h1))
    W6 = np.random.randn(h1, inp) * np.sqrt(2. / h1)
    b6 = np.zeros((1, inp))
    return [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6]

def forward_pass(params, X):

    # # encoder
    # Z1 = np.dot(X, W1) + b1
    # A1 = np.maximum(0, Z1) # use relu function
    # Z2 = np.dot(A1, W2) + b2
    # A2 = (1 / (1 + np.exp(-Z2))) # use sigmoid function

    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6 = params
    Z1 = np.dot(X, W1) + b1
    # apply activation function
    A1 = leaky_relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = leaky_relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = leaky_relu(Z3)
    Z4 = np.dot(A3, W4) + b4
    A4 = leaky_relu(Z4)
    Z5 = np.dot(A4, W5) + b5
    A5 = leaky_relu(Z5)
    Z6 = np.dot(A5, W6) + b6
    A6 = 1 / (1 + np.exp(-Z6)) # sigmoid func
    # return Z1, A1, Z2, A2
    return [Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, Z6, A6]

def back_pass(X_n, X_c, cache, params):
    Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, Z6, A6 = cache
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6 = params
    m = X_c.shape[0]

    dZ6 = A6 - X_c
    dW6 = (1/m) * np.dot(A5.T, dZ6)
    db6 = (1/m) * np.sum(dZ6, axis=0, keepdims=True)

    dA5 = np.dot(dZ6, W6.T)
    dZ5 = dA5 * leaky_relu_deriv(Z5)
    dW5 = (1/m) * np.dot(A4.T, dZ5)
    db5 = (1/m) * np.sum(dZ5, axis=0, keepdims=True)

    dA4 = np.dot(dZ5, W5.T)
    dZ4 = dA4 * leaky_relu_deriv(Z4)
    dW4 = (1/m) * np.dot(A3.T, dZ4)
    db4 = (1/m) * np.sum(dZ4, axis=0, keepdims=True)

    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * leaky_relu_deriv(Z3)
    dW3 = (1/m) * np.dot(A2.T, dZ3)
    db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * leaky_relu_deriv(Z2)
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    # dA1 = np.dot(dZ2, W2.T)
    # # apply relu derivative
    # relu_d = (Z1 > 0).astype(float)
    # dZ1 = dA1 * relu_d
    # dW1 = (1/m) * np.dot(X_n.T, dZ1)
    # db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    # return dW1, db1, dW2, db2

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * leaky_relu_deriv(Z1)
    dW1 = (1/m) * np.dot(X_n.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    return [dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, dW6, db6]

def update_wts(wts, gradient, lr):

    newStuff = []
    for i in range(len(wts)):
        a = wts[i] - (lr * gradient[i])   
        newStuff.append(a)

    return newStuff

def mse(A_final, X_clean):
    return np.mean((A_final - X_clean) ** 2)

# def update_wts(W1, b1, W2, b2, dW1, db1, dW2, db2, a):
#     W1 = W1 - a * dW1
#     b1 = b1 - a * db1
#     W2 = W2 - a * dW2
#     b2 = b2 - a * db2

#     return W1, b1, W2, b2