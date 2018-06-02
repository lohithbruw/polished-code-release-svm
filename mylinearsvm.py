""" Module containing contains the function mylinearsvm which is a custom
    implementation of huberized hinge loss SVM

    The function fit_model is used to train an SVM given X, Y and lambda
"""
import numpy as np


def lhh_sub(y, t, h=0.5):
    """ Huberized hinge loss, piece-wise functions
    """
    yt = y * t
    if yt > 1 + h:
        return 0
    elif abs(1 - yt) <= h:
        return ((1 + h - yt) ** 2) / (4 * h)
    else:
        return 1 - yt



# Vecorize for faster computation
lhh = np.vectorize(lhh_sub)


def obj(beta, X, Y, lamb):
    """ Objective funtion for the Huberized Hinge loss SVM
        return the cost/loss
    """
    reg = np.sum(lamb * np.sum(np.square(beta)))
    n = len(X)
    cost = reg + 1 / n * np.sum(lhh(Y, X.dot(beta)))
    return cost


def lhh_grad(y, X, beta, h=0.5):
    """ Gradient of the piece-wise components for the huberized hinge loss
        return the gradient vector
    """
    yt = y * X.dot(beta)
    if yt > 1 + h:
        grad = np.zeros(beta.shape)
    elif abs(1 - yt) <= h:
        grad = np.array(-(1 + h - yt) * y / (2 * h) * X).reshape(beta.shape)
    else:
        grad = np.array(-y * X).reshape(beta.shape)
    return grad


def computegrad(beta, X, Y, lamb):
    """ Compute the gradient of the huberized hinge loss 
    """
    grad = np.zeros(beta.shape)
    for i in range(len(X)):
        grad += lhh_grad(Y[i], X[i], beta)
    n = len(X)
    return (2 * lamb * beta) + 1 / n * grad


def backtracking(beta, X, Y, lamb, max_iter=20):
    """ Tune the value of the learning rate during gradient descent
    """
    grad_x = computegrad(beta, X, Y, lamb)  # Gradient at x
    norm_grad_x = np.linalg.norm(grad_x)    # Norm of the gradient at x
    found_t = False
    i = 0                                   # Iteration counter
    t = 1
    alpha = 0.5                             # Reasonable default
    b = 0.5                                 # Reasonable default
    while (found_t is False and i < max_iter):
        if (obj(beta - t * grad_x, X, Y, lamb) < obj(beta, X, Y, lamb) - alpha * t * norm_grad_x**2):
            found_t = True
        elif i == max_iter - 1:
            return t
        else:
            t *= b
            i += 1
    return t


def compute_accuracy(beta, X, Y):
    """ Compute the accuracy by predicting y values using X and beta and 
        compare them to the true values (Y)
        return accuracy as a fraction
    """
    num = np.sum((Y * np.dot(X, beta)).ravel() >= 0)
    den = len(Y)
    return num / den


def mylinearsvm(X, Y, lamb, cost_values=None, clf_errors=None, val_errors=None, X_val=None, Y_val=None, max_iter=20):
    """ Learn the values of beta that minimizes the huberized hinge loss
        return the learned betas and optionally the cost, train errors and validation errors
    """
    t = 0
    epsilon = 0.001    # Sensible default
    # Initialize to small values
    beta = np.random.normal(size=X.shape[1]).reshape(-1, 1) * 0.01
    theta = np.zeros(beta.shape)
    grad = None
    for i in range(max_iter):
        if i != 0:                               # in order to capture cost before 1st iteration
            eta = backtracking(beta, X, Y, lamb)
            grad = computegrad(theta, X, Y, lamb)
            beta_n = theta - eta * grad
            theta = beta_n + t / (t + 3) * (beta_n - beta)      # Update rule
            beta = beta_n
            t += 1

        if cost_values is not None:
            cost = obj(beta, X, Y, lamb)
            cost_values.append(cost)

        if clf_errors is not None:
            clf_error = 1 - compute_accuracy(beta, X, Y)
            clf_errors.append(clf_error)

        if val_errors is not None:
            val_errors.append(1 - compute_accuracy(beta, X_val, Y_val))

        if grad is not None and np.linalg.norm(grad) <= epsilon:
            break

    return beta


def fit_model(X_train, Y_train, X_val, Y_val, lamb=0.01):
    """ Fit an SVM given lambda, train and validation data
        return learned betas and performance data
    """
    cost_values = []
    clf_errors = []
    val_errors = []
    betas = mylinearsvm(X_train, Y_train, 0.01, cost_values,
                        clf_errors, val_errors, X_val, Y_val)
    return betas, cost_values, clf_errors, val_errors
