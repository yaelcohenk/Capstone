import pandas as pd
import ray
import optuna
import warnings
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import os
import random
import json

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

    # Aquí le agregué el dropna
    entrenamiento = dataframe_producto[dataframe_producto["year"] < 2023].dropna()
    validacion = dataframe_producto[dataframe_producto["year"] >= 2023].dropna()

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
            ventas_estimadas = None
            
        if ventas_estimadas is not None:
            predicciones = ventas_estimadas.forecast(steps=len(validacion))
            rmse = root_mean_squared_error(
                validacion["Cantidad"].to_numpy(), predicciones)

            if np.isnan(rmse):
                return float("inf")

            return rmse
        else:
            return float("inf")

    objective = partial(optimizar_holt_winters,
                        entrenamiento=entrenamiento, validacion=validacion)
    try:
        valores = study.optimize(objective, n_trials=100)

    except ValueError as e:
        return None, None, None, None, None, validacion.index, nombre_producto
    
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
        predicciones = None

    # Con get_forecast también podemos obtener intervalos de confianza
    if predicciones is not None:
        valores_reales= validacion["Cantidad"].tolist()
        predicciones = predicciones.forecast(steps=len(validacion))
        predicciones1=predicciones.tolist()
        demanda_esc1=[]
        demanda_esc2=[]
        demanda_esc3=[]
    
        
        for i in range (len(predicciones)):
            margen= abs(valores_reales[i]-predicciones[i])
            margen= int(margen)
            numero1= random.randint(-margen, margen)
            demanda1= int(predicciones[i]) + numero1
            numero2= random.randint(-margen, margen)
            demanda2= int(predicciones[i]) + numero2
            numero3= random.randint(-margen, margen)
            demanda3= int(predicciones[i]) + numero3
            if demanda1<0 :
                demanda1=0
            if demanda2<0 :
                demanda2=0
            if demanda3<0:
                demanda3=0
            demanda_esc1.append(demanda1)
            demanda_esc2.append(demanda2)
            demanda_esc3.append(demanda3)
        print(demanda_esc1)
        print(demanda_esc2)
        print(demanda_esc3)
        
            
        
        return predicciones, validacion["Cantidad"].to_numpy(), demanda_esc1, demanda_esc2, demanda_esc3, validacion.index, nombre_producto
    else:
        demanda_esc1=[]
        demanda_esc2=[]
        demanda_esc3=[]
        return None, None, demanda_esc1, demanda_esc2, demanda_esc3, validacion.index, nombre_producto


if __name__ == '__main__':
    ray.init(log_to_driver=False)
    ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos.set_index("Fecha", inplace=True)

    productos = ventas_productos["Descripción"].unique().tolist()
    lista_productos = [ventas_productos[ventas_productos["Descripción"].isin([i])] for i in productos]

    lista_final = list(zip(lista_productos, productos))

    futures = [holt_winters.remote(dataframe_prod, prod)
               for dataframe_prod, prod in lista_final]

    elementos = ray.get(futures)

    dataframes = list()

    valores = list()
    for elemento in elementos:
        predicciones, valor_real, demanda_esc1, demanda_esc2, demanda_esc3, fechas, nombre = elemento
        if predicciones is not None and valor_real is not None:
            mape = mean_absolute_percentage_error(valor_real, predicciones)
            rmse = root_mean_squared_error(valor_real, predicciones)
            mae = mean_absolute_error(valor_real, predicciones)
            mse = mean_squared_error(valor_real, predicciones)
            errores=[]
            suma=0
            for i in range( len(predicciones)):
                error=valor_real[i]-predicciones[i]
                suma+=error
                errores.append(error)
            MAD_prophet=np.mean(np.abs(errores))
            if MAD_prophet!=0:
                tracking_signal=suma/MAD_prophet
            else:
                tracking_signal = np.nan
            valores.append((nombre, mape, rmse, mae, mse,tracking_signal))
        else:
            valores.append((nombre, float("inf"), float("inf"), float("inf"), float("inf"), -1000))


    dataframe = pd.DataFrame(valores)
    dataframe.columns = ["producto", "MAPE", "RMSE", "MAE", "MSE", "tracking_signal"]
    dataframe.set_index("producto", inplace=True)
    dataframe.to_excel(os.path.join("datos", "metricas_holt_winters.xlsx"))
    contador = 0
    mapeo_nombres = dict()
    for data, nombre in dataframes:
        # print(data)


        data.to_excel(os.path.join("predicciones", "holt_winters", f"producto_{contador}.xlsx"))
        
        mapeo_nombres[contador] = nombre
        contador += 1

    with open(os.path.join("predicciones", "holt_winters", "mapeos.txt"), "w") as file:
        json.dump(mapeo_nombres, file)
