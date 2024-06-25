import pandas as pd
import sys
import os
import prophet
import ray
import json
import numpy as np
import random

from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error,
                             mean_absolute_error,
                             mean_squared_error)
from darts import TimeSeries
from darts.models import Croston
@ray.remote
def croston_producto(dataframe_producto: pd.DataFrame, nombre_producto: str):
    datos_validacion_demanda = dataframe_producto[dataframe_producto["ds"].dt.year >= 2023]
    datos_forecasting = dataframe_producto.set_index('ds').asfreq('D', fill_value=0).reset_index()
    datos_validacion = dataframe_producto[dataframe_producto["ds"].dt.year >= 2023]


    datos_forecasting = dataframe_producto[dataframe_producto["ds"].dt.year < 2023]
    series_train = TimeSeries.from_dataframe(datos_forecasting, 'ds', 'y', fill_missing_dates=True, freq='D')
    series_valid = TimeSeries.from_dataframe(datos_validacion, 'ds', 'y', fill_missing_dates=True, freq='D')

    model = Croston(version="optimized")
    model.fit(series_train)

    fechas_futuras = datos_validacion_demanda["ds"].to_frame()



    # print(fechas_futuras)
    forecast = model.predict(len(fechas_futuras))
    predicciones = forecast.values().flatten()
    valores_reales = datos_validacion_demanda["y"]
    lista_fechas = fechas_futuras["ds"].values
    predicciones = predicciones.tolist()
    valores_reales = valores_reales.values.tolist()

    predicciones = [int(i) for i in predicciones]
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

    datos = pd.DataFrame({"Fecha": lista_fechas,"predicciones": predicciones, "real": valores_reales, "escenario1": demanda_esc1, "escenario2": demanda_esc2, "escenario3": demanda_esc3})
    


    mape = mean_absolute_percentage_error(valores_reales, predicciones)
    rmse = root_mean_squared_error(valores_reales, predicciones)
    mae = mean_absolute_error(valores_reales, predicciones)
    mse = mean_squared_error(valores_reales, predicciones)
    errores=[]
    suma=0
    for i in range( len(predicciones)):
        error=valores_reales[i]-predicciones[i]
        suma+=error
        errores.append(error)
    MAD_prophet=np.mean(np.abs(errores))
    if MAD_prophet!=0:
        tracking_signal=suma/MAD_prophet
    else:
        tracking_signal = np.nan


    return nombre_producto, mape, rmse, mae, mse,tracking_signal, datos


if __name__ == '__main__':
    ray.init()
    ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos = ventas_productos[["Fecha", "Cantidad", "Descripción"]]
    ventas_productos["Fecha"] = pd.to_datetime(ventas_productos["Fecha"])
    ventas_productos.columns = ["ds", "y", "Descripción"]
    productos = ventas_productos["Descripción"].unique().tolist()

    lista_productos = [ventas_productos[ventas_productos["Descripción"].isin([i])] for i in productos]
    
    lista_final = list(zip(lista_productos, productos))
    futures = [croston_producto.remote(dataframe_prod, prod) for dataframe_prod, prod in lista_final]

    elementos = ray.get(futures)

    datos_predicciones_productos = list()
    valores = list()


    for elemento in elementos:
        nombre, *metricas, datos = elemento
        valores.append((nombre, *metricas))
        datos_predicciones_productos.append((nombre, datos))

    
    dataframe = pd.DataFrame(valores)
    dataframe.columns = ["producto", "MAPE", "RMSE", "MAE", "MSE", "Tracking Signal"]
    dataframe.set_index("producto", inplace=True)
    dataframe.to_excel(os.path.join("datos", "metricas_croston.xlsx"))

    diccionario_equivalencias_nombres = dict()
    contador = 0
    for nombre_producto, dataframe in datos_predicciones_productos:
        dataframe.to_excel(os.path.join("predicciones", "croston", f"producto_{contador}.xlsx"))
        diccionario_equivalencias_nombres[contador] = nombre_producto
        contador += 1

    with open(os.path.join("predicciones", "croston", "mapeos.txt"), "w") as file:
        json.dump(diccionario_equivalencias_nombres, file)