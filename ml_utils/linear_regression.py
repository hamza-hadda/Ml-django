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
    dataset = pd.read_excel(name, sep=';')
    dataset.describe()

    m = dataset.dropna()
    m.shape

    X = dataset[['Fdw', 'Tdw', 'Ta', 'Thwi', 'Pa', 'RH', 'Frh']]
    y = dataset['Thwo']

    plt.figure(figsize=(15, 10))
    seabornInstance.distplot(dataset['Thwo'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])

    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='red')
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test, y_pred))
