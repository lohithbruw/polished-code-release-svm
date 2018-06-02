from mylinearsvm import fit_model
from misc import print_results
from misc import plot_errors
from misc import get_simulated_data
import random
import numpy as np
import matplotlib.pyplot as plt


def generate_simulated_data(n):
    """
    """
    X = [[random.random() * 3 - 1.5, random.random() * 3 - 1.5]
         for _ in range(n)]
    Y = [1 if b <= a else -1 for a, b in X]
    return np.array(X), np.array(Y).reshape(-1, 1)


def plot_data(X, Y, title):
    x = X[:, 0]
    y = X[:, 1]
    col = ['red' if val == 1 else 'green' for val in Y]
    plt.scatter(x, y, color=col)
    plt.title(title)
    plt.show()


def get_data(train_n=250, val_n=150):
    X_train, Y_train = generate_simulated_data(train_n)
    plot_data(X_train, Y_train, "Training Data")
    
    X_val, Y_val = generate_simulated_data(val_n)
    plot_data(X_val, Y_val, "Validation Data")

    return X_train, Y_train, X_val, Y_val


def run_demo():
    X_train, Y_train, X_val, Y_val = get_data()
    betas, cost_values, clf_errors, val_errors = fit_model(X_train, Y_train, X_val, Y_val)
    plot_errors(cost_values, clf_errors, val_errors)
    print_results(clf_errors, val_errors)



if __name__ == "__main__":
    run_demo()
