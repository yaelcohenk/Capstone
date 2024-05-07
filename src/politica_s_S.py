import pandas as pd
import os
import optuna
import sys
import matplotlib.pyplot as plt
import seaborn as sns



from datetime import timedelta
from functools import partial

optuna.logging.set_verbosity(optuna.logging.WARNING)
datos_test_item = pd.read_excel(os.path.join("datos", "data_items_fillna.xlsx"))
datos_test_venta = pd.read_csv(os.path.join("datos", "data_sales.csv"), sep=";")
datos_test_venta["date"] = pd.to_datetime(datos_test_venta["date"])
# print(datos_test_venta)

datos_2023_2024 = datos_test_venta[datos_test_venta["date"].dt.year >= 2023]
# fecha_min = min(datos_2023_2024["date"])
fecha_max = max(datos_2023_2024["date"])
# Fecha en formato Year, mes, dia
# print(f"Fecha más pequeña: {fecha_min}, tipo {type(fecha_min)}")
# print(f"Fecha más alta: {fecha_max}")

fecha_min = pd.Timestamp(year=2023, month=1, day=1) # Esta tendría que ser la fecha mínima
# print(fecha_2023_inicio)

# sys.exit()
articulo = "pro plan alimento seco para adulto razas medianas 15 kg"

datos_test_item = datos_test_item[datos_test_item["description"].isin([
                                                                      articulo])]
# Pasar todo esto a funciones
precio_venta = datos_test_item["unit_sale_price (CLP)"].to_numpy()[0]
costo_compra = datos_test_item["cost (CLP)"].to_numpy()[0]
costo_almacenar = datos_test_item["storage_cost (CLP)"].to_numpy()[0]
leadtime = int(datos_test_item["leadtime (days)"].to_numpy()[0])
volumen = datos_test_item["size_m3"].to_numpy()[0]
costo_fijo_comprar = datos_test_item["cost_per_purchase"].to_numpy()[0]

item_id = datos_test_item["item_id"].to_numpy()[0]
nombre_prod = datos_test_item["description"].to_numpy()[0]

datos_test_venta["item_id"].replace({item_id: nombre_prod}, inplace=True)

datos_ventas = datos_test_venta[datos_test_venta["item_id"].isin([articulo])]
datos_ventas = datos_ventas[["item_id", "date", "quantity"]]
datos_ventas["date"] = pd.to_datetime(datos_ventas["date"])


datos_ventas = datos_ventas[datos_ventas["date"].dt.year >= 2023]


# Acá en verdad hay que filtrar los datos y ocupar solo los datos del 2023 para adelante
# por como se separaron los datos en el training

# fecha_min = min(datos_ventas["date"]) # Más que la fecha del producto, debería ser la fecha de la primera venta registrada
# fecha_max = max(datos_ventas["date"]) # Lo mismo acá. Eso o tengo que definir de antemano la primera fecha del año y buscar la última de 2024 que se haya vendido


lista_fechas = [fecha_min + timedelta(days=i)
                for i in range((fecha_max - fecha_min).days + 1)]
fechas_ventas = datos_ventas["date"].tolist()
cantidad_ventas = datos_ventas["quantity"].tolist()

diccionario_demandas = dict()

for fecha, cantidad in zip(fechas_ventas, cantidad_ventas):
    diccionario_demandas[fecha] = cantidad


# Hay que tener en cuenta el tema del volumen
def caso_base_T_r_Q(demandas: dict, lista_fechas: list, fecha_min, nombre_prod, T=7, Vmax=10,
                    s=1,
                    S=1):

    contador_dias_pasados = 0
    compras = dict()
    ventas = 0
    ordenes_realizadas = 0
    quiebres_stock = 0
    cantidad_comprada = 0
    demanda_perdida = dict()

    inventario_fechas = {fecha_min - timedelta(days=1): 0}
    lista_fechas_compras = list()
    for fecha in lista_fechas:
        demanda_fecha = diccionario_demandas.get(fecha, 0)

        inventario = inventario_fechas[fecha -
                                       timedelta(days=1)] + compras.get(fecha, 0)

        # print(f"Al inicio del día {fecha} el inventario era {inventario}")
        # print(f"El inventario el día de ayer era {inventario_fechas[fecha - timedelta(days=1)]} y hoy llegaron {compras.get(fecha, 0)} productos")
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
                # print(f"La cantidad a comprar el día {fecha} y que llegará el {fecha + timedelta(days=leadtime)} es de {cantidad_comprar}\n")
                lista_fechas_compras.append(fecha)
                compras[fecha + timedelta(days=leadtime)] = cantidad_comprar
                cantidad_comprada += cantidad_comprar

        contador_dias_pasados += 1
        inventario_fechas[fecha] = inventario
        # print(f"Al final del día {fecha}")
        # print(f"El inventario el día {fecha} es de {inventario}, este día llegaron {compras.get(fecha, 0)} productos comprados y la demanda fue de {demanda_fecha}\n")


    ventas_clp = ventas * precio_venta
    costo_comprar_clp = costo_compra * cantidad_comprada
    costo_fijo_clp = ordenes_realizadas * costo_fijo_comprar
    costo_almacenaje_clp = sum(inventario_fechas.get(
        fecha) * costo_almacenar for fecha in lista_fechas)
    venta_perdida_clp = sum(demanda_perdida.values()) * \
        (precio_venta - costo_compra)

    ganancias = ventas_clp - costo_comprar_clp - \
        costo_fijo_clp - costo_almacenaje_clp - venta_perdida_clp

    print(f"[INFO]: Se vendieron en total de {ventas} unidades")
    print(f"[INFO]: Se hicieron en total {ordenes_realizadas} compras")
    print(f"[INFO]: Se tuvo un total de {quiebres_stock} quiebres de stock")
    print(
        f"[INFO]: La demanda perdida suma un total de {sum(demanda_perdida.values())}")
    print(f"[INFO]: Se compraron en total {cantidad_comprada} productos")
    print(f"[INFO]: Las utilidades corresponden a {ganancias} CLP")

    return ganancias, lista_fechas, list(inventario_fechas.values())[1:], compras, lista_fechas_compras


def caso_base_T_r_Q_optuna(trial,
                           demandas: dict,
                           lista_fechas: list,
                           fecha_min,
                           nombre_prod,
                           T=7,
                           Vmax=10):

    # hacer heurística para ver producto e ir comprando, como según un ranking
    # Nunca nos dará si llenamos la bodega si política está bien calibrada

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

        # print(f"Al inicio del día {fecha} el inventario era {inventario}")
        # print(f"El inventario el día de ayer era {inventario_fechas[fecha - timedelta(days=1)]} y hoy llegaron {compras.get(fecha, 0)} productos")
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
                # print(f"La cantidad a comprar el día {fecha} y que llegará el {fecha + timedelta(days=leadtime)} es de {cantidad_comprar}\n")

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

ganancias, fechas, inventario_caso, compras, fechas_compras = caso_base_T_r_Q(demandas=diccionario_demandas,
                                                lista_fechas=lista_fechas,
                                                fecha_min=fecha_min,
                                                nombre_prod=articulo,
                                                s=best_params["s"],
                                                S=best_params["S"])





fig_contour = optuna.visualization.plot_contour(study, params=["s", "S"])
fig_contour.show()

# print(fechas)


plt.figure(figsize=(20, 6))
sns.lineplot(x=fechas,y=inventario_caso)
plt.xlabel("Fechas")
plt.ylabel("Cantidad de inventario (unidades)")
plt.title(f"Inventario a través timempo del producto X")


# Como que con lo de abajo igual se ve medio feo, pero preguntarle al profe
# for compra in fechas_compras:
    # plt.axvline(x=compra, color = 'r')

# plt.show()
