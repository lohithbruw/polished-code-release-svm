""" This module contains the code to learn a linear SVM on the spam data set.
    It uses the custom implementation of linear SVM.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mylinearsvm import fit_model
from misc import print_results
from misc import plot_errors


def get_data():
    """ Read, clean and return the spam dataset
    """
    spam = pd.read_csv("spam.data", sep=' ', header=None)
    # 57 column contains target (yes/no)
    spam[57] = spam[57].apply(lambda x: -1 if x == 0 else 1)
    # Y now contains 1 and -1
    Y = spam[57].reshape(-1, 1)
    X = spam.drop([57], axis=1)

    # Split the dataset into train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=42)

    # Standardize the data using X_train
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, Y_train, X_val, Y_val


def run_demo():
    """ Sequence of functions to run the demo
    """
    X_train, Y_train, X_val, Y_val = get_data()         # get data
    betas, cost_values, clf_errors, val_errors = fit_model(
        X_train, Y_train, X_val, Y_val)  # fit mode
    plot_errors(cost_values, clf_errors, val_errors)    # plot error curves
    print_results(clf_errors, val_errors)               # print accuracies


if __name__ == "__main__":
    run_demo()
