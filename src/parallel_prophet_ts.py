import pandas as pd
import sys
import os
import prophet
import ray
import json
import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt

from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error,
                             mean_absolute_error,
                             mean_squared_error)

@ray.remote
def prophet_producto(dataframe_producto: pd.DataFrame, nombre_producto: str):
    datos_validacion = dataframe_producto[dataframe_producto["ds"].dt.year >= 2023]
    datos_entrenamiento = dataframe_producto[dataframe_producto["ds"].dt.year < 2023]

    model = prophet.Prophet()
    model.fit(datos_entrenamiento)

    fechas_futuras = datos_validacion["ds"].to_frame()

    forecast = model.predict(fechas_futuras)
    predicciones = forecast["yhat"]
    valores_reales = datos_validacion["y"]
    cota_superior= forecast["yhat_upper"]
    cota_inferior= forecast["yhat_lower"]
    cota_inferior=cota_inferior.values.tolist()
    cota_superior=cota_superior.values.tolist()
    

    lista_fechas = fechas_futuras["ds"].values
    predicciones = predicciones.values.tolist()
    valores_reales = valores_reales.values.tolist()
    
        

    predicciones = [int(i) for i in predicciones]

    


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
    

    
    params = stats.norm.fit(errores)
    fig , ax = plt.subplots()
    v , l , g = ax.hist(errores, bins=20, density = True)
    

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, *params)
    

    num_escenarios = 30
    data1= list()
    esc=list()
    
    escenarios = np.zeros((len(predicciones), num_escenarios))
    for i in range(num_escenarios):
        errores_simulados = stats.norm.rvs(*params, size=len(predicciones))
        escenarios[:, i] = np.round(valores_reales + errores_simulados)
        escenarios[:, i][escenarios[:, i] < 0] = 0



    datos= pd.DataFrame(escenarios, columns=[f'Periodo {i+1}' for i in range(num_escenarios)])
    datos["Fecha"]=lista_fechas
    datos["predicciones"]= predicciones
    datos["real"]= valores_reales


    return nombre_producto, mape, rmse, mae, mse,tracking_signal, datos


if __name__ == '__main__':
    ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos = ventas_productos[["Fecha", "Cantidad", "Descripción"]]
    ventas_productos["Fecha"] = pd.to_datetime(ventas_productos["Fecha"])
    ventas_productos.columns = ["ds", "y", "Descripción"]
    productos = ventas_productos["Descripción"].unique().tolist()

    lista_productos = [ventas_productos[ventas_productos["Descripción"].isin([i])] for i in productos]
    
    lista_final = list(zip(lista_productos, productos))
    futures = [prophet_producto.remote(dataframe_prod, prod) for dataframe_prod, prod in lista_final]

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
    dataframe.to_excel(os.path.join("datos", "metricas_prophet.xlsx"))

    diccionario_equivalencias_nombres = dict()
    contador = 0
    for nombre_producto, dataframe in datos_predicciones_productos:
        dataframe.to_excel(os.path.join("predicciones", "prophet", f"producto_{contador}.xlsx"))
        diccionario_equivalencias_nombres[contador] = nombre_producto
        contador += 1

    with open(os.path.join("predicciones", "prophet", "mapeos.txt"), "w") as file:
        json.dump(diccionario_equivalencias_nombres, file)