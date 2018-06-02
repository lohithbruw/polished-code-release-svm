import matplotlib.pyplot as plt


def print_results(clf_errors, val_errors):
    train_acc = (1 - clf_errors[-1]) * 100
    val_acc = (1 - val_errors[-1]) * 100
    print("Training Accuracy : {:.2f}%".format(train_acc))
    print("Validation Accuracy : {:.2f}%".format(val_acc))


def plot_errors(cost_values, clf_errors, val_errors):
    plt.plot(cost_values)
    plt.title("Cost Value vs Iteration")
    plt.show()

    plt.plot(clf_errors)
    plt.plot(val_errors)
    plt.title("Misclassification Error Rate")
    plt.legend(['Training', 'Validation'])
    plt.show()


def get_simulated_data(train_n=100, val_n=50):
    X_train, Y_train = generate_simulated_data(train_n)
    plot_data(X_train, Y_train, "Training Data")

    X_val, Y_val = generate_simulated_data(val_n)
    plot_data(X_val, Y_val, "Validation Data")

    return X_train, Y_train, X_val, Y_val
