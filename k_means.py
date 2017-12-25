import numpy as np

class K_Means(object):

    def train(self, X_train, K, max_iterations, K_Means_PP=False):  # X_train is an array with shape num_dims x num_data
        """ inputs:
            X_train = data (dimensions: num_dims x num_data)
            K = num_classes
            max_iterations = maximum number of updates
            K_Means_PP = option k-means++ initialisation of centres

            return: None
        """

        self.num_data = X_train.shape[1]
        self.K = K
        self.loss_values = []

        if K_Means_PP == True:
            self.centres = self.improved_initialisation(X_train, K)
        else:
            centres_idx = np.random.choice(range(self.num_data), self.K, replace=False)
            self.centres = X_train[:, centres_idx]

        for i in np.arange(max_iterations):

            # Calculate Euclidean Dist between each data and each centroid
            X_train_extend = X_train.T[:, np.newaxis, :]
            distance_array = np.sum((X_train_extend - self.centres.T)**2, axis=2)

            # 1. Assign points to clusters based on current solution
            self.indicators = np.zeros((self.num_data, self.K))
            self.indicators[range(self.num_data), np.argmin(distance_array, axis=1)] = 1  # argmin returns the column indices

            # Print loss value
            current_loss = np.sum(distance_array[self.indicators == True])
            self.loss_values.append(current_loss)
            print('Iteration: {}, Loss Value: {}'.format(i, current_loss))

            # 2. Update cluster self.centres based on current assignments of points
            self.centres = np.empty(self.centres.shape)
            for j in np.arange(self.K):
                self.centres[:, j] = np.mean(X_train[:, self.indicators[:, j] == 1], axis=1)   # self.indicators[:, j] == 1 returns 1-d array of booleans

        self.labels = np.where(self.indicators == 1)[1]

    def improved_initialisation(self, X_train, K):
        """ return: centres (array) """

        # First centre uniformly from X_train
        centres_idx = np.random.choice(range(X_train.shape[1]), 1, replace=False)
        centres = X_train[:, centres_idx]
        for i in np.arange(K-1):
            # update probability distribution
            prob_dist = self.probability_distribution(X_train, centres)

            # get i-th centre from distribution proportional to d(x)**2
            new_centre_idx = np.random.choice(range(X_train.shape[1]), 1, replace=False, p=prob_dist)  # This returns a list, not just an int
            new_centre = X_train[:, new_centre_idx]
            centres = np.vstack((centres.T, new_centre.T)).T

        return centres

    def probability_distribution(self, X_train, centres):
        """ return: prob_dist (1-d array) """

        # Get distance array
        X_train_extend = X_train.T[:, np.newaxis, :]
        distance_array = np.sum((X_train_extend - centres.T)**2, axis=2)  # dims: num_data x num_classes

        # Construct probability distribution
        prob_dist = distance_array[range(distance_array.shape[0]), np.argmin(distance_array, axis=1)]
        prob_dist = (1/sum(prob_dist)) * prob_dist

        return prob_dist
