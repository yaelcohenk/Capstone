import pandas as pd
import ray
import optuna
import warnings
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging

from parametros import (PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from functools import partial
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error,
                             mean_absolute_error,
                             mean_squared_error)

warnings.filterwarnings('ignore')


@ray.remote
def holt_winters(dataframe_producto: pd.DataFrame, nombre_producto: str):

    study = optuna.create_study(direction="minimize")
    entrenamiento = dataframe_producto[dataframe_producto["year"] < 2023]
    validacion = dataframe_producto[dataframe_producto["year"] >= 2023]

    def optimizar_holt_winters(trial, entrenamiento, validacion):
        smoothing_level = trial.suggest_uniform('alpha', 0, 1)
        smoothing_trend = trial.suggest_uniform('beta', 0, 1)
        smoothing_seasonal = trial.suggest_uniform(
            'seasonal', 1 - smoothing_level, 1)

        try:
            ventas_estimadas = (ExponentialSmoothing(entrenamiento["Cantidad"].to_numpy(),
                                                 trend="add",
                                                 seasonal="add",
                                                 seasonal_periods=7).
                            fit(smoothing_level=smoothing_level,
                                smoothing_trend=smoothing_trend,
                                smoothing_seasonal=smoothing_seasonal))

        except ValueError as e:
            ventas_estimadas = (ExponentialSmoothing(entrenamiento["Cantidad"].to_numpy(),
                                                 trend="add",
                                                 seasonal="add",
                                                 seasonal_periods=2).
                            fit(smoothing_level=smoothing_level,
                                smoothing_trend=smoothing_trend,
                                smoothing_seasonal=smoothing_seasonal))

        predicciones = ventas_estimadas.forecast(steps=len(validacion))
        rmse = root_mean_squared_error(
            validacion["Cantidad"].to_numpy(), predicciones)

        if np.isnan(rmse):
            return float("inf")

        return rmse

    objective = partial(optimizar_holt_winters,
                        entrenamiento=entrenamiento, validacion=validacion)
    valores = study.optimize(objective, n_trials=100)

    best_params = study.best_params
    try:
        # Quizás esto pasarlo a alguna función
        predicciones = (ExponentialSmoothing(entrenamiento["Cantidad"].to_numpy(),
                                         trend="add",
                                         seasonal="add",
                                         seasonal_periods=7).
                    fit(smoothing_level=best_params["alpha"],
                        smoothing_trend=best_params["beta"],
                        smoothing_seasonal=best_params["seasonal"]))

    except ValueError as e:
        predicciones = (ExponentialSmoothing(entrenamiento["Cantidad"].to_numpy(),
                                         trend="add",
                                         seasonal="add",
                                         seasonal_periods=2).
                    fit(smoothing_level=best_params["alpha"],
                        smoothing_trend=best_params["beta"],
                        smoothing_seasonal=best_params["seasonal"]))
    # Con get_forecast también podemos obtener intervalos de confianza
    predicciones = predicciones.forecast(steps=len(validacion))
    return predicciones, validacion["Cantidad"], validacion.index, nombre_producto


if __name__ == '__main__':
    ray.init(log_to_driver=False)
    ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos.set_index("Fecha", inplace=True)

    productos = ventas_productos["Descripción"].unique().tolist()[:3]
    lista_productos = [ventas_productos[ventas_productos["Descripción"].isin([i])] for i in productos]

    lista_final = list(zip(lista_productos, productos))

    futures = [holt_winters.remote(dataframe_prod, prod)
               for dataframe_prod, prod in lista_final]

    elementos = ray.get(futures)


    print(elementos)

    sys.exit("FINAL")
    for elemento in elementos:
        predicciones, valor_real, fechas, nombre = elemento

        print(f"[INFO]: Para el producto {elemento}")
        print(predicciones, valor_real)
        print("\n")



    # ventas_productos = ventas_productos[ventas_productos["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]
    # ventas_productos.drop("Descripción", axis=1, inplace=True)

    # best_params, predicciones, valores_reales, fechas = holt_winters(ventas_productos)

    # datos
    # data = pd.DataFrame({"predicciones": predicciones, "reales": valores_reales})
    # data.set_index(fechas, inplace=True)
#
    # data['reales'].plot(style='b', figsize=(10, 5), label='Original')
    # data['predicciones'].plot(style='r', figsize=(10, 5), label='Predicción')
    # plt.xlabel('Fecha')
    # plt.ylabel('Cantidad')
    # plt.title('Predicciones Holt Winters: Comparación de Serie Original y Predicción')
    # plt.legend()
    # plt.show()
    #  plt.close()

    # print(ventas_productos)
    # print(valores)
    # print(best_params)
    # print(predicciones, valores_reales)


# @ray.remote
# def holt_winter(venta_productos):
    # study = optuna.create_study(direction="minimize")
#
    # pass
#
    # def optimizar_holt_winters(trial, cantidad=venta_productos):
    # smoothing_level = trial.suggest_uniform('alpha', 0, 1)
    # smoothing_trend = trial.suggest_uniform('beta', 0, 1)
    # smoothing_seasonal = trial.suggest_uniform('gamma', 1 - smoothing_level, 1)
#
    # prediccion = ExponentialSmoothing(cantidad,
    #   trend="add",
    #   seasonal="add",
    #   seasonal_periods=7)
#
#
# if __name__ == '__main__':
    # ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    # ventas_productos.index = ventas_productos["Fecha"]
    # ventas_productos.drop("Fecha", axis=1, inplace=True)
#
    # productos = ventas_productos["Descripción"].unique().tolist()
#
    # cantidades_productos = list()
#
    # for producto in productos:
    # producto_loop = ventas_productos[ventas_productos["Descripción"].isin([
    #   producto])]
    # producto_loop = producto_loop["Cantidad"].to_numpy()
    # cantidades_productos.append(producto_loop)
