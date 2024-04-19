import numpy as np
from sklearn.model_selection import train_test_split

## Estimator parameters: all the parameters of an estimator can be set when it is instatiated, and have suitable default values
## You can use Shift+ tab in jupyter to check the possible parameters

X = np.arange(10).reshape((5,2))
y = range(5)

# print("array X, y are: {}, {}".format(X,y))
# print(list(y))


## pass X and y test_size 

X_train, Y_train, y_test, x_test = train_test_split(X, y, test_size=.3)

print(X_train)
print(Y_train)
print("test:")
print(x_test, y_test)

