import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    # TODO
    alpha = 0.01
    epoch = 10000
    for k in range(epoch):
        predict_train = sigmoid(np.matmul(w.T, np.vstack((np.ones((1, N)), X))))
        gradient = np.dot(np.vstack((np.ones((1, N)), X)),(predict_train-y).T)/y.size
        w -= alpha * gradient

    # end answer
    
    return w
