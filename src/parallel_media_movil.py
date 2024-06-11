import pandas as pd
import sys
import os
import json
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error,
                             mean_absolute_error,
                             mean_squared_error)
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
import ray

@ray.remote
def media_movil_producto(dataframe_producto: pd.DataFrame, nombre_producto: str):
    
    dataframe_producto['year_month']=dataframe_producto['ds'].apply(lambda x:pd.Timestamp(x).strftime('%Y-%m'))
    mensual= dataframe_producto.groupby(by=['year_month'])['y'].sum().reset_index()
    mensual["year_month"] = pd.to_datetime(mensual["year_month"])
    mensual.columns = ["ds", "y"]

    
    datos_enteros=mensual
    datos_validacion = mensual[mensual["ds"].dt.year >= 2023 ]
    datos_forecasting = mensual[mensual["ds"].dt.year < 2023]
    largo=len(datos_forecasting)
    
    fechas_futuras = datos_validacion["ds"].to_frame()
    datos_enteros["yhat"]=datos_enteros["y"].rolling(window=3).mean().shift(3)
    test=datos_enteros[largo:]
    
    predicciones = test["yhat"]
    valores_reales = test["y"]
    
    lista_fechas = fechas_futuras["ds"].values
    predicciones = predicciones.values.tolist()
    valores_reales = valores_reales.values.tolist()
    
    predicciones = [int(i) for i in predicciones]
    
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
    


    datos = pd.DataFrame({"predicciones": predicciones, "real": valores_reales})
    
    datos.set_index(lista_fechas, inplace=True)
    



    mape = mean_absolute_percentage_error(valores_reales, predicciones)
    rmse = root_mean_squared_error(valores_reales, predicciones)
    mae = mean_absolute_error(valores_reales, predicciones)
    mse = mean_squared_error(valores_reales, predicciones)


    return nombre_producto, mape, rmse, mae, mse,tracking_signal, datos


if __name__ == '__main__':
    ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES) 
    
    ventas_productos = ventas_productos[["Fecha", "Cantidad", "Descripción"]]
    ventas_productos["Fecha"] = pd.to_datetime(ventas_productos["Fecha"])
    ventas_productos.columns = ["ds", "y", "Descripción"]
    productos = ventas_productos["Descripción"].unique().tolist()

    lista_productos = [ventas_productos[ventas_productos["Descripción"].isin([i])] for i in productos]
    
    lista_final = list(zip(lista_productos, productos))
    futures = [media_movil_producto.remote(dataframe_prod, prod) for dataframe_prod, prod in lista_final]

    elementos = ray.get(futures)

    datos_predicciones_productos = list()
    valores = list()


    for elemento in elementos:
        nombre, *metricas, datos = elemento
        valores.append((nombre, *metricas))
        datos_predicciones_productos.append((nombre, datos))

    
    dataframe = pd.DataFrame(valores)
    dataframe.columns = ["producto", "MAPE", "RMSE", "MAE", "MSE","Tracking signal"]
    dataframe.set_index("producto", inplace=True)
    dataframe.to_excel(os.path.join("datos", "metricas_media_movil.xlsx"))

    diccionario_equivalencias_nombres = dict()
    contador = 0
    for nombre_producto, dataframe in datos_predicciones_productos:
        dataframe.to_excel(os.path.join("predicciones", "media_movil", f"media_movil_producto_{contador}.xlsx"))
        diccionario_equivalencias_nombres[contador] = nombre_producto
        contador += 1

    with open(os.path.join("predicciones", "media_movil", "media_movil_mapeos.txt"), "w") as file:
        json.dump(diccionario_equivalencias_nombres, file)