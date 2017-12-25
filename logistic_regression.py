import numpy as np

class Logistic_Regression(object):
    """ Binary Classifier """

    def train(self, X_train, y_train, eta, lam, iterations):  # Via maximum likelihood
        """ inputs:
            X_train = matrix, columns are data points
            y_train = binary labels, elements are either 1 (for Class 1) or 0 (for Class 2)
            eta = gradient descent step size
            iterations = gradient descent number of iterations
            lam = regularisation constant

            returns: None
        """

        self.num_clss = len(set(y_train))
        self.num_dims = X_train.shape[0]
        self.num_data = X_train.shape[1]
        self.y_train = y_train
        bias_row = np.ones(self.num_data)
        self.X_train = np.vstack((X_train, bias_row))
        self.weights = np.random.rand(self.num_dims + 1,)
        self.gradient_descent(eta, iterations, lam)

    def gradient_descent(self, eta, iter, lam):
        """ returns: None """

        for i in np.arange(iter):
            current_loss = self.loss_function(self.weights, self.X_train, self.y_train, lam)
            # print('Iteration: {}, Loss Value: {}'.format(i, current_loss))
            self.weights = self.weights - eta * self.loss_function_grad(self.weights, self.X_train, self.y_train, lam)

    def loss_function(self, weights, X_train, y_train, lam):
        """ returns: loss value (float) """

        loss = 0
        for i in np.arange(self.num_data):
            h_x = self.h(weights, X_train[:, i])
            y_i = y_train[i]
            loss += (-1 * y_i * np.log(h_x)) - ((1 - y_i) * np.log(1 - h_x))
        reg = lam * np.dot(weights, weights)
        loss = ((1 / self.num_data) * loss) + reg
        return loss

    def loss_function_grad(self, weights, X_train, y_train, lam):
        """ returns: gradient of loss (array) """

        grad = 0
        for i in np.arange(self.num_data):
            grad += (y_train[i] - self.h(weights, X_train[:, i])) * X_train[:, i]
        grad = -(1 / self.num_data) * grad + (lam * 2 * weights)
        return grad

    def h(self, weights, data):
        """ returns: Sigmoid mapping (float) """

        m = np.dot(weights.T, data)
        return 1 / (1 + np.exp(-m))

    def classify(self, X_data):
        """ returns: predictions (array) """

        bias_row = np.ones(X_data.shape[1])
        X_data = np.vstack((X_data, bias_row))
        y_pred = []
        for i in np.arange(X_data.shape[1]):
            data = X_data[:, i]
            if self.h(self.weights, data) > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return np.array(y_pred)
