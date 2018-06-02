from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd

from demo_simulated import get_data
from mylinearsvm import fit_model

def get_accuracy_mylinearsvm(X_train, Y_train, X_val, Y_val):
    betas, cost_values, clf_errors, val_errors = fit_model(X_train, Y_train, X_val, Y_val)
    custom_train_acc = 1 - clf_errors[-1]
    custom_val_acc = 1 - val_errors[-1]
    return custom_train_acc, custom_val_acc
    

def get_accuracy_sklearn(X_train, Y_train, X_val, Y_val):
    clf = LinearSVC()
    clf.fit(X_train, Y_train.ravel())
    train_accuracy = accuracy_score(Y_train, clf.predict(X_train))
    val_accuracy = accuracy_score(Y_val, clf.predict(X_val))
    return train_accuracy, val_accuracy    


def compare_results(custom_train_acc, custom_val_acc, sk_train_acc, sk_val_acc):
    results = pd.DataFrame({'Data Set': ['Train', 'Validation']})
    results['Sklearn'] = [sk_train_acc, sk_val_acc]
    results['Custom'] = [custom_train_acc, custom_val_acc]
    print(results)

def run_demo():
    X_train, Y_train, X_val, Y_val = get_data()
    
    custom_train_acc, custom_val_acc = get_accuracy_mylinearsvm(X_train, Y_train, X_val, Y_val)
    sk_train_acc, sk_val_acc = get_accuracy_sklearn(X_train, Y_train, X_val, Y_val)

    compare_results(custom_train_acc, custom_val_acc, sk_train_acc, sk_val_acc)
    

if __name__ == "__main__":
    run_demo()
