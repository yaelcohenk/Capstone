import os
import sys
import pandas as pd
import ray
import numpy as np
import matplotlib.pyplot as plt


from funcion_rnn import crear_datos_RNN

from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from parametros import (PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)


def rnn(dataframe_producto: pd.DataFrame, nombre_producto: str):
    dataframe = dataframe_producto.drop("Descripción", axis=1)
    dataframe.dropna(inplace=True)

    entrenamiento = dataframe[dataframe["year"] < 2023]
    validacion = dataframe[dataframe["year"] >= 2023]

    columnas_covariables = [
        columna for columna in entrenamiento.columns if columna != "Cantidad"]

    x_train_covariates = entrenamiento[columnas_covariables]
    y_train_value = entrenamiento[["Cantidad"]]

    scaler_trainer = StandardScaler()
    x_train_scaled = scaler_trainer.fit_transform(x_train_covariates)

    scaler_values = StandardScaler()
    y_train_scaled = scaler_values.fit_transform(y_train_value)

    n_future = 1
    n_past = 1

    X_train_shape, Y_train = crear_datos_RNN(n_future, n_past, x_train_scaled, y_train_scaled, entrenamiento)

    X_train, X_test, y_train, y_test = train_test_split(X_train_shape, Y_train,
                                                        test_size=0.2,
                                                        random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=6,
                            activation="relu", input_shape=(1, 12), padding="same"),
        keras.layers.GRU(60, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(50, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(40),
        keras.layers.Dense(1)
    ])

# 
    model.compile(optimizer="adam", loss=MeanSquaredError(),
              metrics=[RootMeanSquaredError()])
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    # Cambiar esto de que estoy probando con el test, es pa visualizar nomás

    columnas_covariables_val = [columna for columna in validacion.columns if columna != "Cantidad"]

    x_train_covariates_val = validacion[columnas_covariables_val]
    y_train_value_val = validacion[["Cantidad"]]

    x_val_scaled = scaler_trainer.transform(x_train_covariates_val)
    y_val_scaled = scaler_values.transform(y_train_value_val)

    X_val, Y_val = crear_datos_RNN(n_future, n_past, x_val_scaled, y_val_scaled, validacion)

    predictions = model.predict(X_val)
    predictions = predictions.squeeze()

    print(validacion.shape)
    print(X_val, Y_val)
    print(Y_val.shape, X_val.shape)

    predictions = scaler_values.inverse_transform(predictions.reshape(-1, 1)).flatten()
    real = scaler_values.inverse_transform(Y_val.reshape(-1, 1)).flatten()

    data = pd.DataFrame({"predictions": predictions, "real": real})


    return data, nombre_producto
    # print(predictions)
    # print(real)
# 
# 
    # plt.plot(data["predictions"], color="red", label="Predicción")
    # plt.plot(data["real"], color="blue", label="Valor Real")
    # plt.title("Predicciones RNN en set de datos validación")
    # plt.legend()
    # plt.show()



if __name__ == '__main__':
    ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos.set_index("Fecha", inplace=True)

    productos = ventas_productos["Descripción"].unique().tolist()[:1]

    lista_productos = [ventas_productos[ventas_productos["Descripción"].isin([i])] for i in productos]
    lista_final = list(zip(lista_productos, productos))

    for dataframe, producto in lista_final:
        rnn(dataframe, producto)
