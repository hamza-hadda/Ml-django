

import pandas as pd
import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics, math
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import  MinMaxScaler




#-----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras 
#-----------------------------------------------------------------------------

# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))







# Importing the dataset
def read_csv_file(path):
    dataset = pd.read_csv(
        path,
        sep=";",
        decimal=","
    )
    x = dataset.iloc[:,:7].values
    y = dataset.iloc[:,7].values
    sc = MinMaxScaler()
    x = sc.fit_transform(x)
    y = y.reshape(-1, 1)
    y = sc.fit_transform(y)
    return dataset, x, y






# Splitting the dataset into the Training set, Test and validation set
def split(x, y, p_x, p_y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=p_x, random_state=4)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=p_y, random_state=4)
    return x_train, x_test, x_val, y_train, y_test, y_val


def ann_model(x_train, y_train, x_test, y_test):
    # built Keras sequential model
    model = Sequential()
    # Adding the input layer:
    model.add(Dense(7, input_dim=x_train.shape[1], activation='relu'))
    # the first hidden layer:
    model.add(Dense(12, activation='relu'))
    # Adding the output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', rmse])

    # enable early stopping based on mean_squared_error
    earlystopping = EarlyStopping(monitor="mean_squared_error", patience=40, verbose=1, mode='auto')

    # Fitting the ANN to the Training set
    result = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, epochs=1000)

    # get predictions
    y_pred = model.predict(x_test)
    metrics_data = {
        "mean_absolute_error": sklearn.metrics.mean_absolute_error(y_test, y_pred),
        "mean_squared_error": sklearn.metrics.mean_squared_error(y_test, y_pred),
        "root_mean_squared_error": math.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)),
        "r_square": sklearn.metrics.r2_score(y_test, y_pred)
    }
    return metrics_data


def ann(path, p_x, p_y):
    dataset, x, y = read_csv_file(path)
    x_train, x_test, x_val, y_train, y_test, y_val = split(x, y, p_x, p_y)
    data = ann_model(x_train, y_train, x_test, y_test)
    return data
