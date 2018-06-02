""" This module contains code that learns an SVM for a simulated dataset
    It uses the custom implementation of linear SVM
"""
import random
import numpy as np
import matplotlib.pyplot as plt

from mylinearsvm import fit_model
from misc import print_results
from misc import plot_errors
from misc import get_simulated_data


def generate_simulated_data(n):
    """ Generate (x, y) co-ordinates within range (-1.5, 1.5)
        set the target to 1 if y <= x else -1
        return the dataset in the form of numpy array
    """
    X = [[random.random() * 3 - 1.5, random.random() * 3 - 1.5]
         for _ in range(n)]
    Y = [1 if b <= a else -1 for a, b in X]
    return np.array(X), np.array(Y).reshape(-1, 1)


def plot_data(X, Y, title):
    """ Use matplot lib to the show the scatter plot of data
        X -> (x, y) the x, y co-ordinates of the observations
        Y -> target : shown in red and blue in the plot
    """
    x = X[:, 0]
    y = X[:, 1]
    col = ['red' if val == 1 else 'green' for val in Y]
    plt.scatter(x, y, color=col)
    plt.title(title)
    plt.show()


def get_data(train_n=250, val_n=150):
    """ Generate simulated data and plot it on 2-D
        Do it for train and validation data
        Return train and validation datasets
    """
    X_train, Y_train = generate_simulated_data(train_n)
    plot_data(X_train, Y_train, "Training Data")

    X_val, Y_val = generate_simulated_data(val_n)
    plot_data(X_val, Y_val, "Validation Data")

    return X_train, Y_train, X_val, Y_val


def run_demo():
    """ Sequence of function calls to run the demo
    """
    X_train, Y_train, X_val, Y_val = get_data()         # get data
    betas, cost_values, clf_errors, val_errors = fit_model(
        X_train, Y_train, X_val, Y_val)  # fit model
    plot_errors(cost_values, clf_errors, val_errors)    # plot the error curves
    print_results(clf_errors, val_errors)               # Print the accuracies


if __name__ == "__main__":
    run_demo()
