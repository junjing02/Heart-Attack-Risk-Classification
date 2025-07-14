############################################################
# NAME       : TING JUN JING
# STUDENT ID : 2300322
############################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------- SECTION 1 ----------------------------------------------------   
def get_splits(df, test_size):
    """
    Task 1.1 Split the dataset to get the training and validation set
    
    Arguments:
    -----------
    - df: panda dataframe
        The dataframe storing the whole dataset.
    - test_size: float with a range of [0, 1]
        The proportion of the dataset to include in the test split.

    Returns:
    ------------
    - X_train: numpy, dtype float64, shape = (M_train, 13)
         The input matrix for training 
    - y_train: numpy, dtype int64, shape = (M_train, 1)
         The targeted variable for training
    - X_val: numpy, dtype float64, shape = (M_val, 13)
         The input matrix for validation
    - y_val: numpy, dtype int64, shape = (M_val, 1)
         The targeted variable for validation
    """
    #####################################################
    # START CODE HERE
    #####################################################
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values.reshape(-1, 1)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size = test_size, 
        random_state = 42
    )
    #####################################################
    # END CODE HERE
    #####################################################

    return X_train, X_val, y_train, y_val


def standardize(X_train, X_val):
    '''
    Task 1.2 Standardize the training set and test set using the mean and standard deviations 
    of the attributes in training set.
    
    Arguments:
    ------------
    - X_train: numpy, dtype float64, shape = (M_train, 13)
         The original input matrix for the training set
    - X_val: numpy, dtype int64, shape = (M_val, 1)
         The original input matrix for the validation set
         
    Returns:
    ------------
    - X_train: numpy, dtype float64, shape = (M_train, 13)
         The standardized input matrix for the training set
    - X_val: numpy, dtype int64, shape = (M_val, 1)
         The standardized input matrix for the validation set
    '''
    #####################################################
    # START CODE HERE
    #####################################################
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    #####################################################
    # END CODE HERE
    #####################################################
    
    return X_train, X_val

# ---------------------------------------------------- SECTION 2 ----------------------------------------------------   
def xavier_init(layer_sizes, rand_seed = 42):
    """
    Task 2.1 Initialize the network parameters
    
    Arguments:
    -----------
    - layer_sizes: list or tuple of integers.
        Stores the dimensions of each layer (including the input layer) in our network.
        For example, when layer_sizes = (2, 10, 5), the input dimension is 2 and there are
        10 neurons for the first layer and 5 for the second layer.
    - rand_seed: integer, default = 42
        The seed for random initialization to ensure reproducibility.

    Returns:
    ------------
    - weights: a list with L+1 numpy arrays 
        Stores the weights initialized with Xavier initialization for all layers. 
        weights[i] is a numpy array storing the weight values for layer i 
        where i = 1...L. For the input layer, weights[0] is always set to None.
    - biases: list with L+1 numpy arrays
        Stores the biases initalized to zeros for all layers. 
        biases[i] is a numpy array storing the bias values
        for layer i where i = 1...L. For the input layer, biases[0] is always set to None.
    """
    np.random.seed(rand_seed)
    L = len(layer_sizes)-1                               # number of layers in the network
    weights = [None for i in range(len(layer_sizes))]    # initialize weights list
    biases  = [None for i in range(len(layer_sizes))]    # initialize biases list
    
    #####################################################
    # START CODE HERE
    #####################################################
    for i in range(1, L + 1):
        F_prev = layer_sizes[i-1]
        F = layer_sizes[i]
        sigma = np.sqrt(2 / (F_prev + F))
        weights[i] = np.random.normal(0, sigma, (F_prev, F))
        biases[i] = np.zeros((F, 1))
    #####################################################
    # END CODE HERE
    #####################################################

    return weights, biases

def forward_linear(A_prev, W, b, act_function):
    """
    Task 2.2 Forward propagation through a linear layer

    Arguments
    ----------
    - A_prev: numpy array of shape (M, F_prev)
        activations from previous layer (or input data).
    - W:  numpy array of shape (F_prev, F)
        The weights matrix for current layer
    - b: numpy array of shape (F, 1)
        The bias vector for current layer
    - act_function: string
        The activation for current layer
        Options: "sigmoid" or "tanh"
    Notes: M: number of samples
           F: number of features (or nodes) in current layer i
           F_prev: number of features (or nodes) in previous layer i-1

    Returns
    ----------
    - A: numpy array of shape (M, F)
        The activation output of current layer
    - cache: a tuple
        Stores W, A_prev and A for backpropagation
    """
    M, F_prev = A_prev.shape
    F, _      = b.shape

    #####################################################
    # START CODE HERE
    #####################################################
    # summation phase 
    Z = np.dot(A_prev, W) + b.T

    # activation phase
    if act_function == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))
    elif act_function == 'tanh':
        A = np.tanh(Z)
    else:
        raise ValueError(f"Unsupported act_function: {act_function}.")

    # create cache 
    cache = (W, A_prev, A)
    #####################################################
    # END CODE HERE
    #####################################################

    return A, cache

def forward (X, weights, biases, act_functions):
    """
    Task 2.3 Forward propagation for the standard neural network

    Arguments:
    ----------
    - X:  numpy array of size (M, Fx)
        Input matrix
    - weights: a list of L+1 numpy arrays
        Stores the weights for all layers. weights[i] stores the weight values for layer i
        where i = 1...L. weights[0] is always set to None
    - biases: a list of L+1 numpy arrays
       Stores the biases for all layers. biases[i] stores the bias value for layer i
       where i = 1...L. Note that biases[0] is set to None
    - act_functions: list of L+1 strings
        The activation functions used in each layer. act_functions[0] is always set to None.
        Supports either 'tanh' or 'sigmoid'
        For example, for a 2-layered NN, we can set act_functions to [None, 'tanh', 'sigmoid']

    Returns:
    -----------
    - y_hat: numpy array of size (M, 1)
        The activation value of the last value or the predicted value
    - caches: list of L+1 items
        Stores the cache generated from the forward_linear for each layer. caches[i] stores
        the cache (tuple) generated by foward propagation for layer i where i = 1...L.
        For the input layer, caches[0] is set to None.
    """
    # number of layers in the neural network
    L = len(weights)-1       

    # initialize caches
    caches = [None for i in range(len(weights))]
    
    #####################################################
    # START CODE HERE
    #####################################################    
    # forward propagation
    A = X
    for l in range(1, L + 1):
        A, cache = forward_linear(A, weights[l], biases[l], act_functions[l])
        caches[l] = cache

    # clip the predicted output for numerical stability
    y_hat = np.clip(A, 1e-5, 1 - 1e-5)
    #####################################################
    # END CODE HERE
    #####################################################

    return y_hat, caches


def compute_cost(y_hat, y_true):
    """
    Task 2.4: Binary cross entropy cost function

    Arguments:
    -----------
    - y_hat: numpy array of shape (M, 1)
        The predicted output which is a probability vector.
    - y_true: numpy array of shape (M, 1)
        The ground truth vector which is a binary vector.

    Returns:
    ---------
    - cost: float
        The cross-entropy cost
    """
    #####################################################
    # START CODE HERE
    #####################################################
    cost = -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))
    #####################################################
    # END CODE HERE
    #####################################################

    return cost    


# ---------------------------------------------------- SECTION 3 ----------------------------------------------------
def backprop_cost_layer(y_hat, y_train):
    """
    Task 3.1 Backpropagation for the cost layer

    Arguments:
    --------------
    - y_hat: numpy array of shape (M, 1)
        The predicted output which is a probability vector.
    - y_true: numpy array of shape (M, 1)
        The ground truth vector which is a binary vector.

    Returns:
    - dAL: numpy array of shape (M, 1)
        Gradient the cost w.r.t. of A[L] or equivalently y_hat
    """
    #####################################################
    # START CODE HERE
    #####################################################
    m = y_hat.shape[0]
    dAL = (((1 - y_train) / (1 - y_hat)) - (y_train / y_hat)) / m
    #####################################################
    # END CODE HERE
    #####################################################
    return  dAL

def backprop_linear_layer(dA, cache, activation_function):
    """
    Task 3.2 Backpropagation for the linear layer

    Arguments:
    --------------
    - dA: numpy array of shape (M, F), the same shape as A
        The upstream gradient for current layer, i.e., the gradient of cost w.r.t. A (i.e., dJ/dA)
    - cache: tuple
        Tuple of values (W, A_prev, A) for the current layer l. The cache was saved during
        forward propagation
    - activation: string
        The activation to be used in this layer.
        Options: "sigmoid" or "tanh"

    Returns:
    - dA_prev: numpy array of shape (M, F_prev), same shape as A_prev
        Gradient the cost w.r.t. of A_prev (i.e., dJ/dA[i-1])
    - dW: numpy array, same shape as W, i.e., (F_prev, F)
        Gradient of the cost w.r.t. W for current layer l (i.e., dJ/dW[i])
    - db: numpy array, same shape as b, i.e., (F, 1)
        Gradient of the cost w.r.t. b for current layer l (i.e., dJ/db[i])
    """
    W, A_prev, A = cache    
    #####################################################
    # START CODE HERE
    #####################################################
    # Backprop through the activation function
    if activation_function == 'sigmoid':
        dZ = dA * (A * (1 - A))
    elif activation_function == 'tanh':
        dZ = dA * (1 - np.square(A))
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}.")

    # backprop through the summation function
    m = A_prev.shape[0]
    dW = A_prev.T.dot(dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True).T / m
    dA_prev = dZ.dot(W.T)
    #####################################################
    # END CODE HERE
    #####################################################

    return dA_prev, dW, db 

def backprop(y_hat, y_true, act_functions, caches):
    """
    Task 3.3 Backpropagation for the standard neural network

    Arguments:
    ------------
    - y_hat: numpy array of shape (M, 1)
        The predicted value, or the output of the forward propagation
    - y_true: numpy array of shape (M, 1)
        The ground truth, or true label (0: negative, 1: positive samples)
    - act_functions: list of shape L+1
        Stores the activation functions used for each layer.
        act_functions[l] stores the activation function ('sigmoid' or 'tanh')
        For layer i, act_functions[0] is always None.
    - caches: list of shape L+1
        Stores the caches for each layer saved during forward propagation.
        cache[l] stores the cache for layer i. cache[0] is always None.

    Returns:
    ------------
    - weights_grad : a list of size L+1
        Stores the gradients of W for all layers, dW[l] for l = 1 ... L. For the input layer, weights_grad[0] is always None.
    - biases_grad : a list of size L+1
        Stores the gradients of b for all layers, db[l] for l = 1 ... L. For the input layer, biases_grad[0] is always None.

    """
    L = len(caches)-1         
    weights_grad = [None for i in range(len(caches))]
    biases_grad  = [None for i in range(len(caches))]

    #####################################################
    # START CODE HERE
    #####################################################
    # Backpropagation for cost function 
    dA = backprop_cost_layer(y_hat, y_true)

    # Backpropagation for layer L to 1
    for l in range(L, 0, -1):

        # get current cache value
        cache = caches[l]

        # compute dA_prev, dW, db by calling "backprop_linear_layer"
        dA_prev, dW, db = backprop_linear_layer(dA, cache, act_functions[l])

        # store dW and db into weights_grad and biases_grad
        weights_grad[l] = dW
        biases_grad[l] = db

        dA = dA_prev  # update dA for the next layer
    #####################################################
    # END CODE HERE
    #####################################################

    return weights_grad, biases_grad


# ---------------------------------------------------- SECTION 4 ----------------------------------------------------   
def train(X_train, y_train, layer_sizes, act_functions, lr, num_iters):
    """
    Task 4.1 Batch Gradient Descent algorithm

    Arguments:
    -------------
    - X_train: ndarray of shape (M, Fx)
        Input matrix for the training set
    - y_train: ndarray of shape (M, 1)
        Targeted variable or ground truth for the training set
    - layer_sizes: list of integers. Size is defined by user, and must be the same as act_functions
        The list containing number of nodes in each layer, including the input layer
        e.g., [30, 50, 1] where 30 is the number of features in each sample, 50 is
        the number of units in the first layer and 1 is the number of nodes in the
        output layer
    - act_functions: tuple of strings. Size is defined by user, and must be the same as layer_sizes
        The list containing the activation functions for each layer
        The first item must be None because there are no activation for the input layer
        e.g., (None, "tanh", "sigmoid")
    - lr: float
        The learning rate for gradient descent. Default = 0.1
    - num_iters: int
        The number of iterations for gradient descent

    Returns:
    -------------
    - weights: a list of L+1 numpy arrays
        Stores the weights for all layers. weights[i] stores the weight values for layer i
        where i = 1...L. weights[0] is always set to None
    - biases: a list of L+1 numpy arrays
       Stores the biases for all layers. biases[i] stores the bias value for layer i
       where i = 1...L. biases[0] is always set to None
    - history: dictionary 
        Stores the training costs (key = 'cost') accross different iteration (key = `iter`).
    """

    np.random.seed(42)
    history = {'iter': [], 'cost': []}

    L = len(layer_sizes) - 1 # number of layers

    #####################################################
    # START CODE HERE
    #####################################################
    # initialize network parameters by calling "xavier_init".
    weights, biases = xavier_init(layer_sizes)

    # repeat until convergence 
    for i in range(num_iters):

        # forward propagation
        y_hat, caches = forward(X_train, weights, biases, act_functions)

        # compute cost 
        cost = compute_cost(y_hat, y_train)

        # backpropagation 
        weights_grad, biases_grad = backprop(y_hat, y_train, act_functions, caches)

        # update network parameters
        for l in range(1, L + 1):
            weights[l] -= lr * weights_grad[l]
            biases[l] -= lr * biases_grad[l]

        # update progress
        if i == 0 or (i+1) % 5 == 0 or i+1 == num_iters:
            history['iter'].append(i)
            history['cost'].append(cost)
            print(f'Iter {i+1:3d} : training cost = {cost:.4f}')
    #####################################################
    # END CODE HERE
    #####################################################
        
    return weights, biases, history


def evaluate(X, y_true, weights, biases, act_functions):
    """
    Task 4.2 Evaluate the trained model on the test set (X, y_true). The model's parameters
    are saved in 'weights' and 'biases' and it uses the activation functions defined in 
    'act_functions'.

    Arguments:
    ------------
    - X: numpy array of shape (M, Fx)
        Input matrix
    - y_true: numpy array of shape (M, 1)
        Output labels (ground truth) for the input matrix X
    - weights: a list of size L+1
        Stores the weights for all layers. weights[i] stores the weight values for layer i
        where i = 1...L. weights[0] is always set to None
    - biases: a list of size L+1
       Stores the biases for all layers. biases[i] stores the bias value for layer i
       where i = 1...L. biases[0] is always set to None
    - act_functions: a list or tuple of size L+1
        The list or tuple containing the activation functions for each layer
        The first item must be None because there are no activation for the input layer
        e.g., (None, "tanh", "sigmoid")

    Returns:
    ------------
    - y_hat: numpy array with shape (M, 1) and dtype float32
        The predicted score by the model
    - y_pred: numpy array with shape (M, 1) and dtype int64
        The predicted score by the model
    - acc:  float
        The accuracy of the model on the dataset
    """
    #####################################################
    # START CODE HERE
    #####################################################
    # Forward propagation
    y_hat, _ = forward(X, weights, biases, act_functions)
    y_hat = y_hat.astype(np.float32)

    # get y_pred
    y_pred = (y_hat >= 0.5).astype(np.int64)

    # compute accuracy
    acc = np.mean(y_pred == y_true)
    #####################################################
    # START CODE HERE
    #####################################################

    return y_hat, y_pred, acc