import pandas as pd
import sys
import os
import prophet
import ray
import json

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

    lista_fechas = fechas_futuras["ds"].values
    predicciones = predicciones.values.tolist()
    valores_reales = valores_reales.values.tolist()

    predicciones = [int(i) for i in predicciones]

    datos = pd.DataFrame({"predicciones": predicciones, "real": valores_reales})
    datos.set_index(lista_fechas, inplace=True)


    mape = mean_absolute_percentage_error(valores_reales, predicciones)
    rmse = root_mean_squared_error(valores_reales, predicciones)
    mae = mean_absolute_error(valores_reales, predicciones)
    mse = mean_squared_error(valores_reales, predicciones)


    return nombre_producto, mape, rmse, mae, mse, datos


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
    dataframe.columns = ["producto", "MAPE", "RMSE", "MAE", "MSE"]
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