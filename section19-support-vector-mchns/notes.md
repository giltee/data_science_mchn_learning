# Support Vector Machines
- SVMs are supervised learning models with associated learning algorithms that analyze data and recognize patterns, used for classification 
- SVM models are a representation of the examples as points in space, mapped so that the examples of separate categories are divided by a clear gap that is as wide as possible. 
- New examples are then mapped into that same space and predicated to belong to a category based on which side of the gap they fall on.  

## Imports
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

```

## Grid Search
- C: controls the cost of misclassifcation, large c value gives low bias and high variance. 
- Gamma: free guasian radio bases function. If gamma is large, variance is small and bias high. If small vice versus. 
- kernel: 