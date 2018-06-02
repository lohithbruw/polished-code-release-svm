from mylinearsvm import fit_model
from misc import print_results
from misc import plot_errors
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings 
warnings = False

def get_data():
    spam = pd.read_csv("spam.data", sep=' ', header=None)
    spam[57] = spam[57].apply(lambda x: -1 if x == 0 else 1)
    Y = spam[57].reshape(-1, 1)
    X = spam.drop([57], axis=1)


    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, Y_train, X_val, Y_val



def run_demo():
    X_train, Y_train, X_val, Y_val = get_data()
    betas, cost_values, clf_errors, val_errors = fit_model(X_train, Y_train, X_val, Y_val)
    plot_errors(cost_values, clf_errors, val_errors)
    print_results(clf_errors, val_errors)


if __name__ == "__main__":
    run_demo()
