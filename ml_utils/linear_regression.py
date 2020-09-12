import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import seaborn as seabornInstance


def linear_regression(name):
    dataset = pd.read_csv(name, sep=';', decimal=',')


    X = dataset[['Fdw', 'Tdw', 'Ta', 'Thwi', 'Pa', 'RH', 'Frl']]
    y = dataset['Thwo']

    plt.figure(figsize=(15, 10))
    seabornInstance.distplot(dataset['Thwo'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    metrics_data = {
        "mean_absolute_error": metrics.mean_absolute_error(y_test, y_pred),
        "mean_squared_error": metrics.mean_squared_error(y_test, y_pred),
        "root_mean_squared_error": np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        "r_square": sklearn.metrics.r2_score(y_test, y_pred)
    }
    return metrics_data
