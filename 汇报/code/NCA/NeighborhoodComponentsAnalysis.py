import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier as kNN, KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def nearest_neighbors(input, label):
    correct = 0
    for i in range(input.shape[0]):
        min_norm = float('inf')
        min_norm_label =  float('inf')
        for j in range(input.shape[0]):
            if i == j:
                continue
            norm = np.linalg.norm(input[i]-input[j])
            if norm < min_norm:
                min_norm = norm
                min_norm_label = label[j]
        if label[i] == min_norm_label:
            correct += 1
    return correct


def scaling_matrix(input):
    input = input.T
    A = np.eye(input.shape[0])
    for i in range(input.shape[0]):
        A[i][i] = 1.0/(max(input[i]) - min(input[i]))
    return A


def scale(ScaleA, input):
    return np.dot(input, ScaleA.T)


def neighborhood_components_analysis(input, label, init, iterations, learning_rate):
    A = init
    A_optimal = init
    correct = 0
    for it in range(iterations):
        i = it % input.shape[0]

        softmax_normalization = 0.
        for k in range(input.shape[0]):
            if k == i:
                continue
            softmax_normalization += np.exp(-np.linalg.norm( np.dot(A, input[i].reshape(-1, 1)) - np.dot(A, input[k].reshape(-1, 1))) )
        if softmax_normalization == 0:
            continue
        softmax = []
        for k in range(input.shape[0]):
            if k == i:
                softmax.append(0.)
            else:
                softmax.append(np.exp(-np.linalg.norm( np.dot(A, input[i].reshape(-1, 1)) - np.dot(A, input[k].reshape(-1, 1))) )/softmax_normalization)
        # print(softmax)
        p = 0.
        for k in range(input.shape[0]):
            if label[k] == label[i]:
                p += softmax[k]
        first_term = np.zeros((input.shape[1], input.shape[1]) )
        second_term = np.zeros((input.shape[1], input.shape[1]) )
        for k in range(input.shape[0]):
            if k == i:
                continue
            xik = input[i] - input[k]
            term = softmax[k] * np.multiply(xik.reshape(-1,1), xik)
            first_term += term
            if label[k] == label[i]:
                second_term += term
        first_term *= p
        A += learning_rate * np.dot(A, (first_term - second_term))
        #if it%(iterations/1000) == 0:
            # print('A\n', A)
            #print('Iteration', it, 'Nearest neighbors on nca data:')
            #nearest_neighbors(scale(A, input), label)
            # plt.scatter(scale(A, X)[:, 0], scale(A, X)[:, 1], c=y)
            # plt.show()
        if it % (iterations / 1000) == 0:
            temp = nearest_neighbors(scale(A, input), label)
            if correct < temp:
                correct = temp
                A_optimal = A
                print('Iteration', it, 'Nearest neighbors on nca data:')
                print('Got', correct, 'correct out of', input.shape[0])
            else:
                print('Iteration', it, 'Nearest neighbors on nca data:')
                print('Got', temp, 'correct out of', input.shape[0])
    return A_optimal


if __name__ == "__main__":
    X, y = load_wine(return_X_y=True)
    # X = np.array([[0, 0, 0.1], [0, 0.1, 0.1], [0.9, 0.6, 0.8], [0.9, 0.5, 0.7]])
    # y = np.array([0, 0, 1, 1])
    print('Nearest neighbors on raw data:')
    print('Got', nearest_neighbors(X, y), 'correct out of', X.shape[0])

    A = scaling_matrix(X)
    print('A\n', A)
    print('Nearest neighbors on scaled data:')
    print('Got', nearest_neighbors(scale(A, X), y), 'correct out of', X.shape[0])

    A = neighborhood_components_analysis(X, y, A, 100000, 0.001)
    print('A\n', A)
    print('Nearest neighbors on nca data:')
    print('Got', nearest_neighbors(scale(A, X), y), 'correct out of', X.shape[0])

    # A = np.array([[-0.09935, -0.2215, 0.3383, 0.443],
    #               [0.2532, 0.5835, - 0.8461, - 0.8915],
    #               [- 0.729, - 0.6386, 1.767, 1.832],
    #               [- 0.9405, - 0.8461, 2.281, 2.794]])
    # nearest_neighbors(scale(A, X), y)