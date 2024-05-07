from functools import partial
from datetime import timedelta
import pandas as pd
import os
import optuna
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import ray

from funciones.parametros_producto import parametros_producto
from politicas.politica_s_S_eval import politica_T_s_S
from politicas.politica_s_S_optuna import politica_T_s_S_optuna
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES


optuna.logging.set_verbosity(optuna.logging.WARNING)


if __name__ == '__main__':
    ray.init()
    datos_items = pd.read_excel(os.path.join("datos", "data_items_fillna.xlsx")) # Tenemos la información de los productos
    datos_ventas = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)


    datos_ventas = datos_ventas[["Fecha", "Cantidad", "Descripción"]]
    datos_ventas = datos_ventas[datos_ventas["Fecha"].dt.year >= 2023]

    fecha_min = pd.Timestamp(year=2023, month=1, day=1)
    fecha_max = max(datos_ventas["Fecha"])
    productos = datos_ventas["Descripción"].unique().tolist()
    

    # Necesito este diccionario
    parametros_producto_modelo = dict()
    
    for producto in productos:
        datos_producto = datos_items[datos_items["description"].isin([producto])]
        *parametros, nombre, ids_asociados = parametros_producto(datos_producto)        
        parametros_producto_modelo[producto] = parametros

    lista_fechas = [fecha_min + timedelta(days=i) for i in range((fecha_max - fecha_min).days + 1)]
    
    # Necesito este diccionario
    fechas_ventas_producto_y_demanda = dict()

    for prod in productos:
        datos_producto_ventas = datos_ventas[datos_ventas["Descripción"].isin([prod])]
        fechas_ventas = datos_producto_ventas["Fecha"].tolist()
        cantidad_ventas = datos_producto_ventas["Cantidad"].tolist()

        diccionario_demandas = dict()


        for fecha, cantidad in zip(fechas_ventas, cantidad_ventas):
            diccionario_demandas[fecha] = cantidad

        fechas_ventas_producto_y_demanda[prod] = diccionario_demandas


    lista_ejecucion_paralelo = list()


    for producto in productos:
        parametros = parametros_producto_modelo[producto]
        demandas = fechas_ventas_producto_y_demanda[producto]

        lista_ejecucion_paralelo.append(politica_T_s_S_optuna.remote(demandas, lista_fechas, fecha_min, producto, parametros))


    resultados = ray.get(lista_ejecucion_paralelo)

    utilidad_total = 0
    ventas_totales = 0
    ordenes_realizadas_total = 0
    quiebres_stock_total = 0
    demanda_perdida_total = 0
    cantidad_comprada_total = 0

    for elementos in resultados:
        utilidad, nombre, ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada = elementos        
        utilidad_total += utilidad
        ventas_totales += ventas
        ordenes_realizadas_total += ordenes_realizadas
        quiebres_stock_total += quiebres_stock
        demanda_perdida_total += demanda_perdida
        cantidad_comprada_total += cantidad_comprada


    print(f"La empresa dentro de todo el período registró utilidad por {utilidad_total} CLP")
    print(f"Se vendieron un total de {ventas_totales} productos")
    print(f"Se emitieron {ordenes_realizadas_total} órdenes de compra")
    print(f"Hubo quiebres de stock en {quiebres_stock_total} veces")
    print(f"La demanda perdida total alcanza el valor de {demanda_perdida_total} unidades")
    print(f"En total se compraron {cantidad_comprada_total} productos")

        # print(f"Para el producto {producto} se obtuvo lo siguiente {elementos}")


