import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import ray
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error,
                             mean_absolute_error,
                             mean_squared_error)
#  root_mean_squared_error)
from sklearn.preprocessing import StandardScaler

from parametros import (FEATURES,
                        TARGET,
                        PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)

# Cada parte lo tengo que dividir en train, test, validación
# Ver si me sirve escalar los datos antes de pasarselo al xgboost
# Preguntar si tenemos que ocupar distintas métricas y de ahí comparar
# Opiniones de los modelos de pronóstico
# Quizás aplicarle un optuna a lo del Regressor


@ray.remote
def xgboost_producto(dataframe_producto: pd.DataFrame,
                     nombre_producto: str = "",
                     random_state=42):

    # print(f"[INFO]: Va a empezar entrenamiento {nombre_producto}")

    try:
        entrenamiento = dataframe_producto[dataframe_producto["year"] < 2023]
        validacion = dataframe_producto[dataframe_producto["year"] >= 2023]

        X_train, y_train = entrenamiento[FEATURES], entrenamiento[TARGET]
        scaler_trainer = StandardScaler()
        scaler_values = StandardScaler()

        x_train_scaled = scaler_trainer.fit_transform(np.array(X_train))
        y_train_scaled = scaler_values.fit_transform(
            np.array(y_train).reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(x_train_scaled, y_train_scaled,
                                                            test_size=0.2,
                                                            random_state=random_state)

        reg = xgb.XGBRegressor(n_estimators=1000, objective="reg:squarederror")

        reg.fit(X_train, y_train, eval_set=[
                (X_train, y_train), (X_test, y_test)], verbose=100)

        X_val, y_val = validacion[FEATURES], validacion[TARGET]
        X_val = scaler_trainer.transform(np.array(X_val))
        y_val = scaler_values.transform(np.array(y_val).reshape(-1, 1))

        # La validación la deberíamos hacer sobre los datos 2023-2024
        prediction = reg.predict(X_val).squeeze()
        prediction = scaler_values.inverse_transform(
            prediction.reshape(-1, 1)).flatten()
        y_val = scaler_values.inverse_transform(y_val.reshape(-1, 1)).flatten()

        # Acá abajo puedo poner más métricas
        mape = mean_absolute_percentage_error(y_val, prediction)
        rmse = root_mean_squared_error(y_val, prediction)
        mae = mean_absolute_error(y_val, prediction)
        mse = mean_squared_error(y_val, prediction)

        return nombre_producto, mape, rmse, mae, mse, y_val, prediction

    except ValueError as e:
        print(f"Ha ocurrido un error {e}")
        return "ValueError"

    # Acá podría retornar todas las métricas de interés, el modelo de regresión, el nombre
    # etc..


if __name__ == '__main__':
    ray.init()
    ventas_productos = pd.read_excel(
        PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos.index = ventas_productos["Fecha"]
    ventas_productos.drop("Fecha", axis=1, inplace=True)
    productos = ventas_productos["Descripción"].unique().tolist()

    # Esto cambiarlo de ahí para ver toods los productos
    # productos = productos[:10]

    lista_productos = [
        ventas_productos[ventas_productos["Descripción"].isin([i])] for i in productos]

    lista_final = list(zip(lista_productos, productos))

    futures = [xgboost_producto.remote(
        dataframe_prod, prod) for dataframe_prod, prod in lista_final]

    # print(ray.get(futures))
    elementos = ray.get(futures)

    valores = list()
    predicciones = list()


    for elemento in elementos:
        # print(elemento[:-2])
        valores.append(elemento[:-2])
        predicciones.append((elemento[0], elemento[-2], elemento[-1]))

    dataframe = pd.DataFrame(valores)
    dataframe.columns = ["producto", "MAPE", "RMSE", "MAE", "MSE"]
    dataframe.set_index("producto", inplace=True)


    # print(dataframe)
    dataframe.to_excel(os.path.join("datos", "metricas_xgboost.xlsx"))
    sys.exit()
    for elemento in predicciones:
        dato_loop = pd.DataFrame({"prediction": elemento[1].tolist(), "real": elemento[2].tolist()})
        print(dato_loop)

        dato_loop['real'].plot(style='b', figsize=(10, 5), label='Original')
        dato_loop['prediction'].plot(style='r', figsize=(10, 5), label='Predicción')

        plt.xlabel('Fecha')
        plt.ylabel('Cantidad')
        plt.title('Predicciones XGBoost: Comparación de Serie Original y Predicción')
        plt.legend()
        plt.show()

        plt.close()