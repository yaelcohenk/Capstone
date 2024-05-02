import pandas as pd
import os
import optuna

from datetime import timedelta
from functools import partial


datos_test_item = pd.read_excel(
    os.path.join("datos", "data_items_fillna.xlsx"))
datos_test_venta = pd.read_csv(
    os.path.join("datos", "data_sales.csv"), sep=";")

articulo = "pro plan alimento seco para adulto razas medianas 15 kg"

datos_test_item = datos_test_item[datos_test_item["description"].isin([
                                                                      articulo])]

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
datos_ventas = datos_ventas[datos_ventas["date"].dt.year > 2023]


# Acá en verdad hay que filtrar los datos y ocupar solo los datos del 2023 para adelante
# por como se separaron los datos en el training

fecha_min = min(datos_ventas["date"])
fecha_max = max(datos_ventas["date"])


lista_fechas = [fecha_min + timedelta(days=i)
                for i in range((fecha_max - fecha_min).days + 1)]
fechas_ventas = datos_ventas["date"].tolist()
cantidad_ventas = datos_ventas["quantity"].tolist()

diccionario_demandas = dict()

for fecha, cantidad in zip(fechas_ventas, cantidad_ventas):
    diccionario_demandas[fecha] = cantidad

def caso_base_T_r_Q(
        demandas: dict,
        lista_fechas: list,
        fecha_min,
        nombre_prod,
        T=7,
        Vmax=10,
        r=1,
        Q=1):

    # r = trial.suggest_int('r', 1, 30)
    # Q = trial.suggest_int('Q', 1, 30)

    # constraint_volumen = y[]
    contador_dias_pasados = 0
    compras = dict()
    ventas = 0
    ordenes_realizadas = 0
    quiebres_stock = 0
    cantidad_comprada = 0
    demanda_perdida = dict()

    inventario = {fecha_min - timedelta(days=1): 0}

    # r, Q = 1, 7
    # print(inventario)

    for fecha in lista_fechas:
        demanda_fecha = diccionario_demandas.get(fecha, 0)

        inventario[fecha] = inventario[fecha -
                                       timedelta(days=1)] + compras.get(fecha - timedelta(days=leadtime), 0)

        if inventario[fecha] < demanda_fecha:
            quiebres_stock += 1
            demanda_perdida[fecha] = demanda_fecha - inventario[fecha]
            ventas += (inventario[fecha])
            inventario[fecha] = 0
        else:
            ventas += demanda_fecha
            inventario[fecha] -= demanda_fecha

        if contador_dias_pasados % T == 0:
            # print("REVISANDO FECHA")
            if inventario[fecha] < r:
                ordenes_realizadas += 1
                compras[fecha] = Q
                cantidad_comprada += Q

        contador_dias_pasados += 1

    ventas_clp = ventas * precio_venta
    costo_comprar_clp = costo_compra * cantidad_comprada
    costo_fijo_clp = ordenes_realizadas * costo_fijo_comprar
    costo_almacenaje_clp = sum(inventario.get(
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

    return ganancias


def caso_base_T_r_Q_optuna(trial,
                           demandas: dict,
                           lista_fechas: list,
                           fecha_min,
                           nombre_prod,
                           T=7,
                           Vmax=10):

    r = trial.suggest_int('r', 1, 30)
    Q = trial.suggest_int('Q', 1, 30)

    # constraint_volumen = y[]
    contador_dias_pasados = 0
    compras = dict()
    ventas = 0
    ordenes_realizadas = 0
    quiebres_stock = 0
    cantidad_comprada = 0
    demanda_perdida = dict()

    inventario = {fecha_min - timedelta(days=1): 0}

    # r, Q = 1, 7
    # print(inventario)

    for fecha in lista_fechas:
        demanda_fecha = diccionario_demandas.get(fecha, 0)

        inventario[fecha] = inventario[fecha -
                                       timedelta(days=1)] + compras.get(fecha - timedelta(days=leadtime), 0)

        constraint_volumen = inventario[fecha] * volumen - Vmax
        trial.set_user_attr("constraint", constraint_volumen)
        if inventario[fecha] < demanda_fecha:
            quiebres_stock += 1
            # Esto es como considerar que vendimos todo lo que teníamos
            # Esto hay que verlo bien y programarlo bien
            # Si nos llegan unidades de demanda y tenemos 5, vendemos las 5?
            # o solo vendemos si es que tenemos lo suficiente para venderles
            # Tiene más sentido vender todo lo que teníamos y de ahí dejar el inventario en cero
            demanda_perdida[fecha] = demanda_fecha - inventario[fecha]
            ventas += (inventario[fecha])
            inventario[fecha] = 0
        else:
            ventas += demanda_fecha
            inventario[fecha] -= demanda_fecha

        if contador_dias_pasados % T == 0:
            # print("REVISANDO FECHA")
            if inventario[fecha] < r:
                ordenes_realizadas += 1
                compras[fecha] = Q
                cantidad_comprada += Q

        contador_dias_pasados += 1

    ventas_clp = ventas * precio_venta
    costo_comprar_clp = costo_compra * cantidad_comprada
    costo_fijo_clp = ordenes_realizadas * costo_fijo_comprar
    costo_almacenaje_clp = sum(inventario.get(fecha) * costo_almacenar for fecha in lista_fechas)
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

study.optimize(objective, n_trials=1000)
best_params = study.best_params
best_value = study.best_value
print("Optimized parameters:", best_params)
print("Optimized function value:", best_value)


print(f"[INFO]: Para {best_params} se obtiene")

caso_base_T_r_Q(demandas=diccionario_demandas,
                lista_fechas=lista_fechas,
                fecha_min=fecha_min,
                nombre_prod=articulo,
                r=best_params["r"],
                Q=best_params["Q"])
