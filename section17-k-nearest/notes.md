# K-Nearest Neighbour 
- Classifcation algorithm that operates on a simple principle.

Training Algorithm:
1. Store all the data

Prediction Algorithm: 
1. Calculate the distance from x to all points in your data
2. Sort the points in your data by increasing distance from x 
3. Predict the majority label of the k closest points

Pros:
- simple
- training is trivial
- works with any number of classes
- easy to add more data
- few parameters
    - K 
    - Distance metric

Cons:
- High prediction cost, worse for large data sets
- Not good with high dimensional data
- Categorical Features don't work well

## Common Interview Task
- You're given anonymized data and attempt to classify it, without knowing the context of the data. 
- We will simulate this scenario using KNN to classify it. 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

%matplotlib inline

## Choosing a K value
- take a range (loop) and get the predictions and error rate (np.mean(pred_i != y_test))