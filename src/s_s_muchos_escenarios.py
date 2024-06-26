from functools import partial
from datetime import timedelta
import pandas as pd
import os
import optuna
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import ray
import pickle
import json
import numpy as np

from funciones.parametros_producto import parametros_producto
from politicas.politica_s_S_eval import politica_T_s_S
from politicas.politica_s_S_optuna import politica_T_s_S_optuna
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
from funciones.calcular_cosas_politicas import calcular_metricas

optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna.logging.disable_propagation()

if __name__ == '__main__':
    np.random.seed(2123)
    ray.init(log_to_driver=True)

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

    lista_ejecucion_paralelo = dict()


    # print(parametros_producto_modelo)

    with open("diccionario_params_productos.pkl", "wb") as f:
        pickle.dump(parametros_producto_modelo, f)

    with open("distrs.pkl", "rb") as file:
        distrs = pickle.load(file)

        # print(distrs)

        # sys.exit()
    # sys.exit()
    k = 100
    escenarios = list()
    

    for escenario in range(k):
        
        valores_escenario_k = list()
        
        for producto in productos:
            parametros = parametros_producto_modelo[producto]
            demandas = fechas_ventas_producto_y_demanda[producto]

            distr_producto = distrs[producto]
            mu, std = distr_producto


            demanda_perturbada = dict()

            for fecha, demanda_dia in demandas.items():
                
                demanda_perturbada[fecha] = max(0, int(demanda_dia + np.random.normal(mu, std)))


            valores_escenario_k.append(politica_T_s_S_optuna.remote(demanda_perturbada, lista_fechas, fecha_min, producto, parametros))

        escenarios.append(valores_escenario_k)

    columns = ["utilidad_total", "ventas_totales", "ordenes_realizadas_total", "quiebres_stock_total",
               "demanda_perdida_total", "cantidad_comprada_total", "costo_alm_total", "nivel_rotacion"]
    dataframe_data = pd.DataFrame(columns=columns)
    
    print("[INFO]: A punto de correr varios escenarios: ")

    for num_escenario, escenario in enumerate(escenarios):
        resultados = ray.get(escenario)
        print(f"Para el escenario {num_escenario} se tienen las siguientes métricas")
        valores_loop = calcular_metricas(resultados, lista_fechas, leadtime_on=False)


        dataframe_data.loc[len(dataframe_data)] = valores_loop
        if num_escenario % 5 == 0:
            dataframe_data.to_excel("s_s_pronosticos_escenarios.xlsx")


        # print(dataframe_data)

    dataframe_data.to_excel("s_s_pronosticos_escenarios.xlsx")

    sys.exit()
    utilidad_total = 0
    ventas_totales = 0
    ordenes_realizadas_total = 0
    quiebres_stock_total = 0
    demanda_perdida_total = 0
    cantidad_comprada_total = 0
    costo_alm_total = 0
    costo_productos_vendidos = 0
    inventario_prom = 0

    inventarios_productos = dict()

    for elementos in resultados:
        utilidad, nombre, ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada, inv, costo_alm_prod, costo_fijo_clp, costo_compra_clp = elementos        
        utilidad_total += utilidad
        ventas_totales += ventas
        ordenes_realizadas_total += ordenes_realizadas
        quiebres_stock_total += quiebres_stock
        demanda_perdida_total += demanda_perdida
        cantidad_comprada_total += cantidad_comprada
        costo_alm_total += costo_alm_prod

        costo_productos_vendidos += costo_fijo_clp
        costo_productos_vendidos += costo_compra_clp
        
        

        inventarios_productos[nombre] = inv
        inventario_fechas=inventarios_productos[nombre]
        i=0
        cantidad=0
        for fecha in lista_fechas:
            cantidad += int(inventario_fechas[fecha])
            i+=1
    
        inventario_prom += cantidad/i 

    nivel_rotacion = costo_productos_vendidos/inventario_prom
    print(f"La empresa dentro de todo el período registró utilidad por {utilidad_total} CLP")
    print(f"Se vendieron un total de {ventas_totales} productos")
    print(f"Se emitieron {ordenes_realizadas_total} órdenes de compra")
    print(f"Hubo quiebres de stock en {quiebres_stock_total} veces")
    print(f"La demanda perdida total alcanza el valor de {demanda_perdida_total} unidades")
    print(f"En total se compraron {cantidad_comprada_total} productos")
    print(f"El costo de almacenaje total fue de {costo_alm_total}")
    print(f"El nivel de rotación es de {nivel_rotacion}")

    mapeo_graficos = dict()
    contador = 0

    # inventarios_ = dict()

    for producto, inventario in inventarios_productos.items():
        plt.figure(figsize=(16, 6))
        sns.lineplot(x=lista_fechas, y=list(inventario.values())[1:])
        plt.xlabel("Fechas")
        plt.ylabel("Cantidad de inventario (unidades)")
        plt.title(f"Inventario a través del tiempo para {producto}")
        plt.savefig(os.path.join("politicas_graficos", "inventario", "t_s_S", f"prod_{contador}.png"))
        plt.close()

        mapeo_graficos[contador] = producto
        contador += 1

        # print(f"Para el producto {producto} se obtuvo lo siguiente {elementos}")
    with open(os.path.join("politicas_graficos", "inventario", "t_s_S", "mapeos.txt"), "w") as file:
        json.dump(mapeo_graficos, file)


    inventario_sistema = np.zeros(shape=len(lista_fechas))

    # si esto no me funciona, uso reduce
    for inventario_prod in inventarios_productos.values():
        inventario_valores = list(inventario_prod.values())[1:]
        numpy_inv = np.array(inventario_valores)
        inventario_sistema =  inventario_sistema.astype(int) + numpy_inv.astype(int)

    plt.figure(figsize=(16, 6))
    sns.lineplot(x=lista_fechas, y=inventario_sistema)
    plt.xlabel("Fechas")
    plt.ylabel("Cantidad de inventario (unidades)")
    plt.title(f"Inventario a través del tiempo para el sistema")
    plt.savefig(os.path.join("politicas_graficos", "inventario", "t_s_S", f"inventario_sistema.png"))
    plt.close()
    