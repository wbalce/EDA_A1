import numpy as np
from collections import Counter

class K_Nearest_Neighbours(object):

    def train(self, X_matrix, y_train):
        """ inputs:
            X_matrix = array of data (dimensions: num_dims x num_data)
            y_train = 1d array of labels

            return: None
        """

        self.X_train = X_matrix.T
        self.y_train = y_train

    def classify(self, dists, k):
        """ inputs:
            dists = distance array
            k = number of neighbours to consider

            return: array of label predictions
        """

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in np.arange(num_test):
            nearest_y = []
            dists_row = dists[i]  # Get row of distances (corresponds with test data i)
            dists_idx = dists_row.argsort()  # Sort row of distances between test data i and all training data
            nearest_y = np.take(self.y_train, dists_idx[0:k])
            nearest_y_list = nearest_y.tolist()
            nearest_y_list.sort()
            y_pred[i] = Counter(nearest_y_list).most_common(1)[0][0]
        return y_pred

    def l2_dist(self, X):
        """ return: distance array (distances between every test data to every training data) """

        X = X.T
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        X_vect = np.sum(X * X, axis=1)
        X_matx = np.tile(X_vect, (num_train,1))  # Matrix with X_vect as repeating columns
        X_train_vect = np.sum(self.X_train * self.X_train, axis=1)
        X_train_matx = np.tile(X_train_vect, (num_test,1))
        cross_Terms = np.dot(X, self.X_train.T)
        dists = np.sqrt(X_matx.T + X_train_matx - 2 * cross_Terms)
        return dists
