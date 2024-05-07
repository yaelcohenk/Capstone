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
datos_test_item = pd.read_excel(os.path.join("datos", "data_items_fillna.xlsx"))
datos_test_venta = pd.read_csv(os.path.join("datos", "data_sales.csv"), sep=";")
datos_test_venta["date"] = pd.to_datetime(datos_test_venta["date"])

datos_2023_2024 = datos_test_venta[datos_test_venta["date"].dt.year >= 2023]

fecha_min = pd.Timestamp(year=2023, month=1, day=1)
fecha_max = max(datos_2023_2024["date"])

# Esta tendría que ser la fecha mínima


articulo = "pro plan alimento seco para adulto razas medianas 15 kg"

datos_test_item = datos_test_item[datos_test_item["description"].isin([
                                                                      articulo])]
# Pasar todo esto a funciones
precio_venta, costo_compra, costo_almacenar, leadtime, volumen, costo_fijo_comprar, item_id, nombre_prod, *others = parametros_producto(
    datos_test_item)

datos_test_venta["item_id"].replace({item_id: nombre_prod}, inplace=True)

datos_ventas = datos_test_venta[datos_test_venta["item_id"].isin([articulo])]
datos_ventas = datos_ventas[["item_id", "date", "quantity"]]
datos_ventas["date"] = pd.to_datetime(datos_ventas["date"])


datos_ventas = datos_ventas[datos_ventas["date"].dt.year >= 2023]

lista_fechas = [fecha_min + timedelta(days=i) for i in range((fecha_max - fecha_min).days + 1)]
fechas_ventas = datos_ventas["date"].tolist()
cantidad_ventas = datos_ventas["quantity"].tolist()

diccionario_demandas = dict()

for fecha, cantidad in zip(fechas_ventas, cantidad_ventas):
    diccionario_demandas[fecha] = cantidad

def caso_base_T_r_Q_optuna(trial,
                           demandas: dict,
                           lista_fechas: list,
                           fecha_min,
                           nombre_prod,
                           T=7,
                           Vmax=10):
    
    s = trial.suggest_int('s', 1, 30)
    S = trial.suggest_int('S', s, 30)

    contador_dias_pasados = 0
    compras = dict()
    ventas = 0
    ordenes_realizadas = 0
    quiebres_stock = 0
    cantidad_comprada = 0
    demanda_perdida = dict()

    inventario_fechas = {fecha_min - timedelta(days=1): 0}
    for fecha in lista_fechas:
        demanda_fecha = diccionario_demandas.get(fecha, 0)

        inventario = inventario_fechas[fecha -
                                       timedelta(days=1)] + compras.get(fecha, 0)

        if inventario < demanda_fecha:
            quiebres_stock += 1
            demanda_perdida[fecha] = demanda_fecha - inventario
            ventas += (inventario)
            inventario = 0
        else:
            ventas += demanda_fecha
            inventario -= demanda_fecha

        if contador_dias_pasados % T == 0:
            if inventario < s:
                ordenes_realizadas += 1
                cantidad_comprar = (S - inventario)

                compras[fecha + timedelta(days=leadtime)] = cantidad_comprar
                cantidad_comprada += cantidad_comprar

        contador_dias_pasados += 1
        inventario_fechas[fecha] = inventario

    ventas_clp = ventas * precio_venta
    costo_comprar_clp = costo_compra * cantidad_comprada
    costo_fijo_clp = ordenes_realizadas * costo_fijo_comprar
    costo_almacenaje_clp = sum(inventario_fechas.get(
        fecha) * costo_almacenar for fecha in lista_fechas)
    venta_perdida_clp = sum(demanda_perdida.values()) * \
        (precio_venta - costo_compra)

    ganancias = ventas_clp - costo_comprar_clp - \
        costo_fijo_clp - costo_almacenaje_clp - venta_perdida_clp

    return ganancias


study = optuna.create_study(direction="maximize")
objective = partial(caso_base_T_r_Q_optuna,
                    demandas=diccionario_demandas,
                    lista_fechas=lista_fechas,
                    fecha_min=fecha_min,
                    nombre_prod=articulo)

study.optimize(objective, n_trials=100)
best_params = study.best_params
best_value = study.best_value
print("Optimized parameters:", best_params)
print("Optimized function value:", best_value)


print(f"[INFO]: Para {best_params} se obtiene")




ganancias, fechas, inventario_caso, compras, fechas_compras = politica_T_s_S(diccionario_demandas=diccionario_demandas,
                                                                             lista_fechas=lista_fechas,
                                                                             fecha_min=fecha_min,
                                                                             nombre_prod="",
                                                                             leadtime=leadtime,
                                                                             precio_venta=precio_venta,
                                                                             costo_fijo_comprar=costo_fijo_comprar,
                                                                             costo_compra=costo_compra,
                                                                             costo_almacenar=costo_almacenar,
                                                                             s=best_params["s"],
                                                                             S=best_params["S"])


fig_contour = optuna.visualization.plot_contour(study, params=["s", "S"])
# fig_contour.show()

# print(fechas)


# plt.figure(figsize=(20, 6))
# sns.lineplot(x=fechas, y=inventario_caso)
# plt.xlabel("Fechas")
# plt.ylabel("Cantidad de inventario (unidades)")
# plt.title(f"Inventario a través timempo del producto X")

# Como que con lo de abajo igual se ve medio feo, pero preguntarle al profe
# for compra in fechas_compras:
# plt.axvline(x=compra, color = 'r')

# plt.show()


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


    for producto in productos[:3]:
        parametros = parametros_producto_modelo[producto]
        demandas = fechas_ventas_producto_y_demanda[producto]

        lista_ejecucion_paralelo.append(politica_T_s_S_optuna.remote(demandas, lista_fechas, fecha_min, producto, parametros))


    resultados = ray.get(lista_ejecucion_paralelo)
    print(resultados)

