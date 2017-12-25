
import numpy as np
import logistic_regression as logreg
import itertools
from collections import Counter

class One_vs_One(object):

    def train(self, X_train, y_train, eta, lam, iterations):
        """ inputs:
            X_train = matrix, columns are data points
            y_train = binary labels, elements are either 1 (for Class 1) or 0 (for Class 2)
            eta = gradient descent step size
            itererations = gradient descent number of iterations
            lam = regularisation constant

            returns: None
        """

        self.num_pairs = len(set(y_train))
        self.classifiers = {}  # { class pair : classifier for pair }

        # Build pair_list
        for pair in itertools.combinations(set(y_train), 2):
            self.classifiers.update({pair : None})

        for pair in self.classifiers:
            logreg_obj = logreg.Logistic_Regression()

            # Set up data
            y_train_bin = y_train[np.logical_or(y_train == pair[0], y_train == pair[1])]
            y_train_bin[y_train_bin == pair[0]] = 1  # Class 1 -> 1, Since pair = (Class 1, Class 2)
            y_train_bin[y_train_bin == pair[1]] = 0  # Class 2 -> 0
            X_train_bin = X_train[:, np.logical_or(y_train == pair[0], y_train == pair[1])]

            logreg_obj.train(X_train_bin, y_train_bin, eta, lam, iterations)
            self.classifiers[pair] = logreg_obj

    def classify(self, X_data):
        """ returns: class predictions (array) """

        Y_pred = np.zeros(X_data.shape[1])  # Initialise Y_pred with dummy row
        y_pred_final = []
        for pair in self.classifiers:
            y_pred = self.classifiers[pair].classify(X_data)  # array
            y_pred[y_pred == 1] = pair[0]   # 1 -> Class 1
            y_pred[y_pred == 0] = pair[1]   # 0 -> Class 2
            Y_pred = np.vstack((Y_pred, y_pred))
        np.delete(Y_pred, 0, axis=0)  # Delete dummy row

        # Extract most common elements in column
        for col in np.arange(Y_pred.shape[1]):
            obj_counter = Counter(Y_pred[:, col].tolist())
            y_pred_final.append(obj_counter.most_common(1)[0][0])

        return np.array(y_pred_final)
