import optuna
import os
import sys
import ray

from datetime import timedelta
from functools import partial
from politicas.r_q_fieles_eval import politica_T_r_Q


@ray.remote
def politica_T_r_Q_optuna(demandas: dict, lista_fechas: list, fecha_min, nombre_prod, params_producto, es_fiel):
    study = optuna.create_study(direction="maximize")

    precio_venta, costo_compra, costo_almacenar, leadtime, volumen, costo_fijo_comprar = params_producto

    def optimizar_t_r_Q_optuna(trial,
                               diccionario_demandas,
                               lista_fechas,
                               fecha_min,
                               nombre_prod,
                               leadtime,
                               precio_venta,
                               costo_fijo_comprar,
                               costo_compra,
                               costo_almacenar,
                               es_fiel,
                               T=7):

        r = trial.suggest_int('r', 1, 30)
        Q = trial.suggest_int('Q', 1, 30)

        contador_dias_pasados = 0
        compras = dict()
        ventas = 0
        ordenes_realizadas = 0
        quiebres_stock = 0
        cantidad_comprada = 0
        demanda_perdida = dict()
        quiebres_stock_fiel = 0

        orden_en_camino = False

        inventario_fechas = {fecha_min - timedelta(days=1): 0}
        for fecha in lista_fechas:
            demanda_fecha = diccionario_demandas.get(fecha, 0)

            inventario = inventario_fechas[fecha -
                                           timedelta(days=1)] + compras.get(fecha, 0)

            if compras.get(fecha, 0) != 0:
                orden_en_camino = False

            if inventario < demanda_fecha:
                quiebres_stock += 1
                demanda_perdida[fecha] = demanda_fecha - inventario
                ventas += (inventario)
                inventario = 0
                if es_fiel:
                    quiebres_stock_fiel += 1
            else:
                ventas += demanda_fecha
                inventario -= demanda_fecha

            if contador_dias_pasados % T == 0:
                if inventario < r and orden_en_camino is False:
                    ordenes_realizadas += 1
                    cantidad_comprar = Q

                    compras[fecha +
                            timedelta(days=leadtime)] = cantidad_comprar
                    cantidad_comprada += cantidad_comprar

                    orden_en_camino = True

            contador_dias_pasados += 1
            inventario_fechas[fecha] = inventario

        ventas_clp = ventas * precio_venta
        costo_comprar_clp = costo_compra * cantidad_comprada
        costo_fijo_clp = ordenes_realizadas * costo_fijo_comprar
        costo_almacenaje_clp = sum(inventario_fechas.get(fecha) * costo_almacenar for fecha in lista_fechas)
        venta_perdida_clp = sum(demanda_perdida.values()) * (precio_venta - costo_compra)

        ganancias = ventas_clp - costo_comprar_clp - costo_fijo_clp - costo_almacenaje_clp - venta_perdida_clp

        return ganancias

    objective = partial(optimizar_t_r_Q_optuna,
                        diccionario_demandas=demandas,
                        lista_fechas=lista_fechas,
                        fecha_min=fecha_min,
                        nombre_prod=nombre_prod,
                        leadtime=leadtime,
                        precio_venta=precio_venta,
                        costo_fijo_comprar=costo_fijo_comprar,
                        costo_compra=costo_compra,
                        costo_almacenar=costo_almacenar,
                        es_fiel=es_fiel)
    
    study.optimize(objective, n_trials=10) # Cambiar número de trials a un número menor quizás
    best_params = study.best_params


    ganancias, *others, ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada, inv, costo_almacenaje_prod, costo_fijo_clp, costo_compra_clp, quiebres_stock_fiels = politica_T_r_Q(diccionario_demandas=demandas,
                                                                             lista_fechas=lista_fechas,
                                                                             fecha_min=fecha_min,
                                                                             nombre_prod="",
                                                                             leadtime=leadtime,
                                                                             precio_venta=precio_venta,
                                                                             costo_fijo_comprar=costo_fijo_comprar,
                                                                             costo_compra=costo_compra,
                                                                             costo_almacenar=costo_almacenar,
                                                                             es_fiel=es_fiel,
                                                                             r=best_params["r"],
                                                                             Q=best_params["Q"])

    return ganancias, nombre_prod, ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada, inv, costo_almacenaje_prod, costo_fijo_clp, costo_compra_clp, quiebres_stock_fiels



@ray.remote
def politica_T_r_Q_optuna_sensibilidad(demandas: dict, lista_fechas: list, fecha_min, nombre_prod, params_producto, lista_leadtimes, es_fiel):
    study = optuna.create_study(direction="maximize")

    precio_venta, costo_compra, costo_almacenar, leadtime, volumen, costo_fijo_comprar = params_producto

    def optimizar_t_r_Q_optuna(trial,
                               diccionario_demandas,
                               lista_fechas,
                               fecha_min,
                               nombre_prod,
                               leadtime,
                               precio_venta,
                               costo_fijo_comprar,
                               costo_compra,
                               costo_almacenar,
                               es_fiel,
                               T=7):

        r = trial.suggest_int('r', 1, 30)
        Q = trial.suggest_int('Q', 1, 30)

        contador_dias_pasados = 0
        compras = dict()
        ventas = 0
        ordenes_realizadas = 0
        quiebres_stock = 0
        cantidad_comprada = 0
        demanda_perdida = dict()
        quiebres_stock_fiel = 0

        orden_en_camino = False

        inventario_fechas = {fecha_min - timedelta(days=1): 0}
        for fecha in lista_fechas:
            demanda_fecha = diccionario_demandas.get(fecha, 0)

            inventario = inventario_fechas[fecha -
                                           timedelta(days=1)] + compras.get(fecha, 0)

            if compras.get(fecha, 0) != 0:
                orden_en_camino = False

            if inventario < demanda_fecha:
                quiebres_stock += 1
                demanda_perdida[fecha] = demanda_fecha - inventario
                ventas += (inventario)
                inventario = 0
                if es_fiel:
                    quiebres_stock_fiel += 1
            else:
                ventas += demanda_fecha
                inventario -= demanda_fecha

            if contador_dias_pasados % T == 0:
                if inventario < r and orden_en_camino is False:
                    ordenes_realizadas += 1
                    cantidad_comprar = Q

                    compras[fecha +
                            timedelta(days=leadtime)] = cantidad_comprar
                    cantidad_comprada += cantidad_comprar

                    orden_en_camino = True

            contador_dias_pasados += 1
            inventario_fechas[fecha] = inventario

        ventas_clp = ventas * precio_venta
        costo_comprar_clp = costo_compra * cantidad_comprada
        costo_fijo_clp = ordenes_realizadas * costo_fijo_comprar
        costo_almacenaje_clp = sum(inventario_fechas.get(fecha) * costo_almacenar for fecha in lista_fechas)
        venta_perdida_clp = sum(demanda_perdida.values()) * (precio_venta - costo_compra)

        ganancias = ventas_clp - costo_comprar_clp - costo_fijo_clp - costo_almacenaje_clp - venta_perdida_clp

        return ganancias

    objective = partial(optimizar_t_r_Q_optuna,
                        diccionario_demandas=demandas,
                        lista_fechas=lista_fechas,
                        fecha_min=fecha_min,
                        nombre_prod=nombre_prod,
                        leadtime=leadtime,
                        precio_venta=precio_venta,
                        costo_fijo_comprar=costo_fijo_comprar,
                        costo_compra=costo_compra,
                        costo_almacenar=costo_almacenar,
                        es_fiel = es_fiel)
    
    study.optimize(objective, n_trials=100) # Cambiar número de trials a un número menor quizás
    best_params = study.best_params

    valores = dict()
    contador = 0

    # print(lista_leadtimes)
    for loop_leadtime in lista_leadtimes:
        ganancias, *others, ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada, inv, costo_almacenaje_prod, costo_fijo_clp, costo_compra_clp, quiebres_stock_fiel = politica_T_r_Q(diccionario_demandas=demandas,
                                                                             lista_fechas=lista_fechas,
                                                                             fecha_min=fecha_min,
                                                                             nombre_prod="",
                                                                             leadtime=loop_leadtime,
                                                                             precio_venta=precio_venta,
                                                                             costo_fijo_comprar=costo_fijo_comprar,
                                                                             costo_compra=costo_compra,
                                                                             costo_almacenar=costo_almacenar,
                                                                             es_fiel=es_fiel, 
                                                                             r=best_params["r"],
                                                                             Q=best_params["Q"])
        

        # print(indice)}
        # print(ganancias, nombre_prod, ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada, inv, costo_almacenaje_prod, costo_fijo_clp, costo_compra_clp)
        valores[contador] = (ganancias, nombre_prod, ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada, inv, costo_almacenaje_prod, costo_fijo_clp, costo_compra_clp, quiebres_stock_fiel)
        contador +=1

    # print(valores)
    return valores
    # return ganancias, nombre_prod, ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada, inv, costo_almacenaje_prod, costo_fijo_clp, costo_compra_clp