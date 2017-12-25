import numpy as np
import scipy.io as sio
import k_nearest_neighbours as knn
import linear_regression as lr
import one_vs_one as ovo
import k_means as km
import cv2
import matplotlib.pyplot as plt

data = sio.loadmat('data/MNIST_engn8535.mat')
y_train = data['trnY'][0]  # Class labels
X_train = data['trnX']  # Columns represent 28x28 images, vectorised
y_test = data['tstY'][0]
X_test = data['tstX']

num_dims = X_train.shape[0]

# ==============================================================================
# Question 1a: K NEAREST NEIGHBOURS, effect of value k
# ==============================================================================
def kNN():
    K = 50
    accuracy_list = []

    for k in np.arange(1, K + 1):
        accuracy_per_fold = []
        X_and_y = np.vstack((X_train, y_train))  # Temporarily stack into one array to shuffle
        np.random.shuffle(X_and_y.T)
        X_train_shuffled = X_and_y[0: num_dims, :]
        y_train_shuffled = X_and_y[num_dims, :]
        X_train_split = np.hsplit(X_train_shuffled, 10)  # A list of equal sized sub arrays
        y_train_split = np.hsplit(y_train_shuffled, 10)
        for s in np.arange(10):
            X_test_fold = X_train_split.pop()
            y_test_fold = y_train_split.pop()
            X_train_fold = np.concatenate(tuple(X_train_split), axis=1)
            y_train_fold = np.concatenate(tuple(y_train_split))  # This is an array

            # ==================================================================
            # kNN Workflow
            obj = knn.K_Nearest_Neighbours()
            obj.train(X_train_fold, y_train_fold)
            dist_matx = obj.l2_dist(X_test_fold)
            y_pred_fold = obj.classify(dist_matx, k)
            accuracy = sum(y_test_fold == y_pred_fold) / y_test_fold.shape[0]
            accuracy_per_fold.append(accuracy)
            # ==================================================================

            # Return popped element back into list of subsets
            X_train_split.append(X_test_fold)
            y_train_split.append(y_test_fold)

        accuracy_for_k = np.mean(accuracy_per_fold)
        accuracy_list.append(accuracy_for_k)
        print('k = {}, accuracy = {}'.format(k, accuracy_for_k))
    x_range = np.arange(1, len(accuracy_list) + 1)
    plt.plot(x_range, accuracy_list, 'r-')
    plt.xlabel('K Values')
    plt.ylabel('Accuracy')
    plt.savefig('kNN.png')

    # Nearest Neighbour Accuracy
    obj_nn = knn.K_Nearest_Neighbours()
    obj_nn.train(X_train, y_train)
    dist_matx_nn = obj_nn.l2_dist(X_test)
    y_pred_nn = obj_nn.classify(dist_matx_nn, 1)
    accuracy_nn = sum(y_test == y_pred_nn) / y_test.shape[0]
    print('K = 1, Accuracy = ', accuracy_nn)

    # Best K Accuracy
    best_k = np.argmax(accuracy_list) + 1
    obj_best_k = knn.K_Nearest_Neighbours()
    obj_best_k.train(X_train, y_train)
    dist_matx_best_k = obj_best_k.l2_dist(X_test)
    y_pred_best_k = obj_best_k.classify(dist_matx_best_k, best_k)
    accuracy_best_k = sum(y_test == y_pred_best_k) / y_test.shape[0]
    print('Best K = {}, Accuracy = {}'.format(best_k, accuracy_best_k))

# ==============================================================================
# Question 1b: LINEAR REGRESSION, Bias vs No Bias
# ==============================================================================
def linear_regression():
    obj_1 = lr.Linear_Regression()
    obj_1.train(X_train, y_train, 0.0001, 100)  # 0.00001 best so far.
    y_pred_1 = obj_1.classify(X_train)
    accuracy_1 = sum(y_test == y_pred_1) / y_test.shape[0]
    print('accuracy (no bias) =', accuracy_1 * 100)

    obj_2 = lr.Linear_Regression()
    obj_2.train(X_train, y_train, 0.0001, 100, True)
    y_pred_2 = obj_2.classify(X_train)
    accuracy_2 = sum(y_test == y_pred_2) / y_test.shape[0]
    print('accuracy (bias) =', accuracy_2 * 100)

# ==============================================================================
# Question 1d: LOGISTIC REGRESSION (ONE VS ONE), best regularisation coefficient
# ==============================================================================
def logistic_regression_ovo():
    accuracy_dict = {}
    for l in np.arange(-5, 2):
        accuracy_per_fold = []
        X_and_y = np.vstack((X_train, y_train))  # Temporarily stack into one array to shuffle
        np.random.shuffle(X_and_y.T)
        X_train_shuffled = X_and_y[0: num_dims, :]
        y_train_shuffled = X_and_y[num_dims, :]
        X_train_split = np.hsplit(X_train_shuffled, 10)  # A list of equal sized sub arrays
        y_train_split = np.hsplit(y_train_shuffled, 10)

        for s in np.arange(10):
            X_test_fold = X_train_split.pop()
            y_test_fold = y_train_split.pop()
            X_train_fold = np.concatenate(tuple(X_train_split), axis=1)
            y_train_fold = np.concatenate(tuple(y_train_split))  # This is an array

            # ==================================================================
            # Logistic Regression (One vs One) Workflow
            obj_ovo = ovo.One_vs_One()
            obj_ovo.train(X_train_fold, y_train_fold, 0.1, 10**l, 100)
            y_pred_fold = obj_ovo.classify(X_test_fold)
            accuracy = sum(y_test_fold == y_pred_fold) / y_test_fold.shape[0]
            accuracy_per_fold.append(accuracy)
            # ==================================================================

            # Return popped element back into list of subsets
            X_train_split.append(X_test_fold)
            y_train_split.append(y_test_fold)

        accuracy_for_l = np.mean(accuracy_per_fold)
        accuracy_dict.update({accuracy_for_l : l})
        print('lambda = {}, accuracy = {}'.format(l, accuracy_for_l))

    best_l = accuracy_dict[max(accuracy_dict)]
    obj_ovo_final = ovo.One_vs_One()
    obj_ovo_final.train(X_train, y_train, 0.1, 10**best_l, 100)
    y_pred = obj_ovo_final.classify(X_test)
    accuracy = sum(y_test == y_pred) / y_test.shape[0]
    print('Best lambda = {}, Accuracy = {}'.format(best_l, accuracy))

# ==============================================================================
# Question 3a: K-MEANS VS K-MEANS++ (SEGMENTATION)
# ==============================================================================
def seg_comparison():
    img = cv2.imread('data/peppers.bmp')
    dim = img.shape
    img = img.reshape(-1,3).T
    K = 10
    colours = ['b', 'r', 'g']
    for k in np.arange(2, 5):
        print('K =', k)
        obj_1 = km.K_Means()
        obj_2 = km.K_Means()
        obj_1.train(img, k, 40)
        obj_2.train(img, k, 40, True)
        img_1 = obj_1.centres[:, obj_1.labels].T.reshape(dim)
        img_2 = obj_2.centres[:, obj_2.labels].T.reshape(dim)
        cv2.imwrite('img_kmeans_k' + str(k) + '.jpg', img_1)
        cv2.imwrite('img_kmeanspp_k' + str(k) + '.jpg', img_2)

        x_range = np.arange(len(obj_1.loss_values))
        colour = colours[k - 2]
        plt.plot(x_range, obj_1.loss_values, colour + '-', label='km, K='+str(k))
        plt.plot(x_range, obj_2.loss_values, colour + '--', label='km++, K='+str(k))

    plt.xlabel('K-Means Iteration')
    plt.ylabel('Loss Value')
    plt.legend(loc='best')
    plt.savefig('km_vs_kmpp.png')
