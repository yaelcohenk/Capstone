import optuna
import os
import sys
import ray

from datetime import timedelta
from functools import partial
from .politica_s_S_eval import politica_T_s_S


@ray.remote
def politica_T_s_S_optuna(demandas: dict, lista_fechas: list, fecha_min, nombre_prod, params_producto):
    study = optuna.create_study(direction="maximize")

    precio_venta, costo_compra, costo_almacenar, leadtime, volumen, costo_fijo_comprar = params_producto

    def optimizar_t_s_S_optuna(trial,
                               diccionario_demandas,
                               lista_fechas,
                               fecha_min,
                               nombre_prod,
                               leadtime,
                               precio_venta,
                               costo_fijo_comprar,
                               costo_compra,
                               costo_almacenar,
                               T=7):

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

                    compras[fecha +
                            timedelta(days=leadtime)] = cantidad_comprar
                    cantidad_comprada += cantidad_comprar

            contador_dias_pasados += 1
            inventario_fechas[fecha] = inventario

        ventas_clp = ventas * precio_venta
        costo_comprar_clp = costo_compra * cantidad_comprada
        costo_fijo_clp = ordenes_realizadas * costo_fijo_comprar
        costo_almacenaje_clp = sum(inventario_fechas.get(fecha) * costo_almacenar for fecha in lista_fechas)
        venta_perdida_clp = sum(demanda_perdida.values()) * (precio_venta - costo_compra)

        ganancias = ventas_clp - costo_comprar_clp - costo_fijo_clp - costo_almacenaje_clp - venta_perdida_clp

        return ganancias

    objective = partial(optimizar_t_s_S_optuna,
                        diccionario_demandas=demandas,
                        lista_fechas=lista_fechas,
                        fecha_min=fecha_min,
                        nombre_prod=nombre_prod,
                        leadtime=leadtime,
                        precio_venta=precio_venta,
                        costo_fijo_comprar=costo_fijo_comprar,
                        costo_compra=costo_compra,
                        costo_almacenar=costo_almacenar)
    
    study.optimize(objective, n_trials=100) # Cambiar número de trials a un número menor quizás
    best_params = study.best_params

    ganancias, *others, ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada, inv, costo_almacena_prod, costo_fijo_clp, costo_compra_clp = politica_T_s_S(diccionario_demandas=demandas,
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

    return ganancias, nombre_prod, ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada, inv, costo_almacena_prod, costo_fijo_clp, costo_compra_clp