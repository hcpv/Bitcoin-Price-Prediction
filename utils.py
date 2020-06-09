from datetime import datetime
from sklearn.metrics import mean_squared_error as mse
import math


def convert(X):
    return [datetime.fromtimestamp(x) for x in X]


def error(y_test, y_pred):
    return math.sqrt(mse(y_test, y_pred))
