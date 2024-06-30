import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from fieles.prod_clientes_fieles import productos_comprados_por_clientes_fieles

df_clientes = pd.read_csv(os.path.join("datos", "data_sales.csv"), sep=";")

def obtener_suma_cantidad_por_item_id_ordenado(df_clientes, productos_comprados_por_clientes_fieles):
    resultados_por_anio = {}
    
    # Obtener productos comprados por clientes fieles por año
    for anio in range(2020, 2024):
        productos_por_anio = productos_comprados_por_clientes_fieles(df_clientes, anio)
        resultados_por_anio[anio] = productos_por_anio

    # Crear un diccionario para almacenar la suma de la cantidad por item_id
    suma_cantidad_por_item_id = {}

    # Iterar sobre cada año en resultados_por_anio
    for anio, productos in resultados_por_anio.items():
        # Excluir el año 2024
        if (anio != 2024):
            # Iterar sobre cada fila (producto) en el DataFrame de productos
            for _, producto in productos.iterrows():
                # Obtener el item_id y la cantidad comprada
                item_id = producto['item_id']
                cantidad = producto['quantity']
                # Sumar la cantidad al total acumulado para el item_id actual
                suma_cantidad_por_item_id[item_id] = suma_cantidad_por_item_id.get(item_id, 0) + cantidad

    # Ordenar el diccionario por valor (cantidad total) en orden decreciente
    suma_cantidad_por_item_id_ordenado = dict(sorted(suma_cantidad_por_item_id.items(), key=lambda item: item[1], reverse=True))

    # Crear un diccionario para almacenar la cantidad de veces que se compró cada producto por año
    conteo_compras_por_item_id = {}

    # Iterar sobre cada año en resultados_por_anio
    for anio, productos in resultados_por_anio.items():
        # Excluir el año 2024
        if anio != 2024:
            # Iterar sobre cada fila (producto) en el DataFrame de productos
            for _, producto in productos.iterrows():
                # Obtener el item_id
                item_id = producto['item_id']
                # Incrementar el contador de compras para el item_id actual
                conteo_compras_por_item_id[item_id] = conteo_compras_por_item_id.get(item_id, 0) + 1

    # Filtrar los productos que se compraron más de una vez en cada año
    productos_frecuentes_por_anio = {item_id: cantidad for item_id, cantidad in conteo_compras_por_item_id.items() if cantidad > 1}

    # Crear un diccionario para almacenar la suma de la cantidad por item_id
    suma_cantidad_por_item_id = {}

    # Iterar sobre cada año en resultados_por_anio
    for anio, productos in resultados_por_anio.items():
        # Excluir el año 2024
        if anio != 2024:
            # Iterar sobre cada fila (producto) en el DataFrame de productos
            for _, producto in productos.iterrows():
                # Obtener el item_id y la cantidad comprada
                item_id = producto['item_id']
                cantidad = producto['quantity']
                # Verificar si el item_id es frecuente (se compró más de una vez en el año)
                if item_id in productos_frecuentes_por_anio:
                    # Sumar la cantidad al total acumulado para el item_id actual
                    suma_cantidad_por_item_id[item_id] = suma_cantidad_por_item_id.get(item_id, 0) + cantidad

    # Ordenar el diccionario por valor (cantidad total) en orden decreciente
    suma_cantidad_por_item_id_ordenado = dict(sorted(suma_cantidad_por_item_id.items(), key=lambda item: item[1], reverse=True))

    return suma_cantidad_por_item_id_ordenado
