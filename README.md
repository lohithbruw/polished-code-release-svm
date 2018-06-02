# mylinearsvm

This code implements a linear SVM with huberized hinge loss. The Project contains 3 demo files
  - demo_simulated.py : Learns an SVM for a simulated dataset
  - demo_real_world.py : Learns an SVM for a real world data set (Spam data set)
  - demo_comparison.py Compares the performance of the Custom implementation of SVM and the sklearn implementation of LinearSVC

# Requirements

  - python3
  - numpy, pandas, sklearn

## Run demo_simulated.py
```python
python demo_simulated.py
```
- Visualize the train and validation data
- Visualize the train performance
- Compute the train and validation accuracy

## Run demo_real_world.py
```python
python demo_real_world.py
```
- Get the spam dataset from https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data
- Clean the dataset
- Visualize the train performance in terms of misclassification error for train and validation set
- Compute the train and validation accuracy for the spam data set


## Run demo_comparison.py
```python
python demo_comparison.py
```
- Generate the simulated dataset
- Learn an SVM model using the custom implementation
- Learn an SVM model using the sklearn implementation of LinearSVC
- Compute and compare the train and accuracy scores for both the models


