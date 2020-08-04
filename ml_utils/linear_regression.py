import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import seaborn as seabornInstance



# In[2]:


def linear_regression(name):
    dataset = pd.read_excel(name, sep=';')
    dataset.describe()

    # In[5]:

    m = dataset.dropna()
    m.shape

    # In[6]:

    X = dataset[['Fdw', 'Tdw', 'Ta', 'Thwi', 'Pa', 'RH', 'Frh']]
    y = dataset['Thwo']

    # In[7]:

    plt.figure(figsize=(15, 10))
    seabornInstance.distplot(dataset['Thwo'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # In[9]:

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # In[14]:

    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    coeff_df

    # In[10]:

    y_pred = regressor.predict(X_test)

    # In[11]:

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1

    # In[12]:

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='red')
    plt.show()

    # In[13]:

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test, y_pred))


# In[ ]: