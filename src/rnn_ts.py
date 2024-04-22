import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
from sklearn.model_selection import train_test_split


ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
ventas_productos.index = ventas_productos["Fecha"]
ventas_productos.drop("Descripción", axis=1, inplace=True)
ventas_productos.drop("Fecha", axis=1, inplace=True)

print(ventas_productos.shape)

ventas_productos.dropna(inplace=True)
print(ventas_productos.shape)

# sys.exit()


columnas_covariables = [columna for columna in ventas_productos.columns if columna != "Cantidad"]

x_train_covariates = ventas_productos[columnas_covariables]
y_train_value = ventas_productos[["Cantidad"]]


scaler_trainer = StandardScaler()
x_train_scaled = scaler_trainer.fit_transform(x_train_covariates)

scaler_values = StandardScaler()
y_train_scaled = scaler_values.fit_transform(y_train_value)

X_train = []
Y_train = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 7  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(x_train_scaled) - n_future + 1):
    X_train.append(x_train_scaled[i - n_past:i, 0:ventas_productos.shape[1]])
    Y_train.append(y_train_scaled[i + n_future - 1:i + n_future, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

Y_train = Y_train.squeeze()

# print(X_train.shape, Y_train.shape)


X_train_shape = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
# print(X_train_shape.shape, Y_train.shape)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(7, 11)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))


model.compile(optimizer="adam", loss="mse")
model.fit(X_train_shape, Y_train, epochs=10)

