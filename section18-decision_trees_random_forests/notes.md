# Decision Trees and Random Forests
- Read Chapter 8 
- Decision trees start from a root node and traverse down to the leaf nodes based on the boolean value of the decision
- To improve performance, we can use many trees with a random sample of features chosen as the split
    - A new random sample of features is chosen for **every single tree at every single split**
    - For **classification**, m is typically chosen to be the square root of p. p is full set of features
- Why use?
    - Suppose there is one very strong feature in the data set. When using "bagged" trees, most of the trees will use that feature as the top split, resulting in an ensemble of similar trees that are highly correlated. 
    - averaging highly correlated quantities doesnt reduce variance by much
    - By randomly leaving out candidate features from each split **Random Forests 'decorrelates' the trees**, such that the averaging process can reduce the variance of the resulting model


## imports
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# sklearn libs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
```


## Tree visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
