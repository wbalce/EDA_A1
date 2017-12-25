import numpy as np
from numpy import linalg as la
import scipy.optimize as opt

class Linear_Regression(object):

    def train(self, X_train, Y_train, eta, iterations, bias=False):
        """ returns: None """

        self.bias = bias
        self.num_clss = len(set(Y_train))
        self.num_dims = X_train.shape[0]
        self.num_data = X_train.shape[1]
        if self.bias == True:
            bias_row = np.ones(self.num_data)
            X_train = np.vstack((X_train, bias_row))
            self.num_dims += 1

        Y_train = np.tile(Y_train, (self.num_clss,1))
        self.Weights = np.random.rand(self.num_dims, self.num_clss)
        self.gradient_descent(X_train, Y_train, eta, iterations)

    def gradient_descent(self, X_train, Y_train, eta, iter):
        for i in np.arange(iter):
            current_loss = self.loss_function(self.Weights, X_train, Y_train)
            # print('Iteration: {}, Loss Value: {}'.format(i, current_loss))
            self.Weights = self.Weights - (eta * self.loss_function_grad(self.Weights, X_train, Y_train))

    def loss_function(self, Weights, X, Y):
        A = np.dot(Weights.T, X) - Y
        B = np.sum(A**2, axis=1)
        return 0.5 * B

    def loss_function_grad(self, Weights, X, Y):
        A = np.dot(np.dot(X, X.T), Weights)
        B = np.dot(X, Y.T)
        return (A - B)

    def classify(self, X):
        if self.bias == True:
            bias_row = np.ones(self.num_data)
            X = np.vstack((X, bias_row))

        linear_transformation = np.dot(self.Weights.T, X)
        classification = np.argmax(linear_transformation, axis=0)
        return classification
