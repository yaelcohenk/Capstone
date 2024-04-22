import pandas as pd
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from tensorflow import keras
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)


def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np) - window_size):
    row = [[a] for a in df_as_np[i: i + window_size]]
    X.append(row)
    label = df_as_np[i + window_size]
    y.append(label)
  return np.array(X), np.array(y)

ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
ventas_productos = ventas_productos[ventas_productos["Descripción"].isin(
    ["pro plan alimento seco para adulto razas medianas 15 kg"])]

ventas_productos.index = ventas_productos["Fecha"]
ventas_productos.drop("Fecha", axis=1, inplace=True)
ventas_productos.drop("Descripción", axis=1, inplace=True)

train, test = train_test_split(ventas_productos, test_size=0.2, random_state=1)

# test, validation = train_test_split(test, test_size=0.25, random_state=1)

# print(train)

# print(test)


# sys.exit()
# print(train.shape, test.shape, validation.shape)

scaler = MinMaxScaler()
df_for_training_scaled = scaler.fit_transform(train)
df_for_testing_scaled= scaler.transform(test)

def createXY(dataset, n_past):
    dataX = []
    dataY = []
    
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)


def df_to_X_y2(df, window_size):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i + window_size][0]
    y.append(label)
  return np.array(X), np.array(y)


window_size = 5


# X_train, y_train = createXY(df_for_training_scaled, window_size)
X_train, y_train = df_to_X_y2(train, window_size)
X_test, y_test = df_to_X_y2(test, window_size)




print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = keras.models.Sequential()
model.add(keras.layers.InputLayer((X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dense(8, 'relu'))
model.add(keras.layers.Dense(1, 'linear'))

model.summary()

model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[RootMeanSquaredError()])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

# sys.exit()
# print(df_for_testing_scaled)
# print(df_for_training_scaled)

# sys.exit()

# print("trainX Shape-- ",trainX.shape)
# print("trainY Shape-- ",trainY.shape)
# 
# 
# print("testX Shape-- ",testX.shape)
# print("testY Shape-- ",testY.shape)

# sys.exit()

FEATURES = ['dayofyear',
            'dayofweek',
            'quarter',
            'month',
            'year',
            'weekofyear',
            "diff_tiempo_venta",
            "fourier_transform",
            "trend", "seasonal"]

TARGET = "Cantidad"

# X_train, y_train = train[FEATURES], train[TARGET]
# 
# X_test, y_test = test[FEATURES], test[TARGET]

# X_valid, y_valid = validation[FEATURES], validation[TARGET]

# number_of_features = len(X_train.columns)

# print(X_train.shape, X_test.shape)

# print(trainX)


# sys.exit()





# model = keras.models.Sequential([
    # keras.layers.GRU(20, input_shape=[window_size, num_features]),
    # keras.layers.GRU(20),
    # keras.layers.TimeDistributed(keras.layers.Dense(1))
# ])
# 
# print(model.summary())
# 
# model.compile(loss=MeanSquaredError(), optimizer="adam", metrics=[RootMeanSquaredError()])
# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
# 
# model.evaluate(X_valid, y_valid)
# 
# plot_learning_curves(history.history["loss"], history.history["val_loss"])
# plt.show()