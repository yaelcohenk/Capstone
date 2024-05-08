from datetime import timedelta


def politica_T_r_Q(diccionario_demandas: dict,
                   lista_fechas: list,
                   fecha_min,
                   nombre_prod,
                   leadtime,
                   precio_venta,
                   costo_fijo_comprar,
                   costo_compra,
                   costo_almacenar,
                   T=7,
                   Vmax=10,
                   r=1,
                   Q=1):

    contador_dias_pasados = 0
    compras = dict()
    ventas = 0
    ordenes_realizadas = 0
    quiebres_stock = 0
    cantidad_comprada = 0
    demanda_perdida = dict()

    orden_en_camino = False

    inventario_fechas = {fecha_min - timedelta(days=1): 0}
    lista_fechas_compras = list()
    for fecha in lista_fechas:
        demanda_fecha = diccionario_demandas.get(fecha, 0)

        inventario = inventario_fechas[fecha - timedelta(days=1)] + compras.get(fecha, 0)

        if compras.get(fecha, 0) != 0:
            orden_en_camino = False

        if inventario < demanda_fecha:
            quiebres_stock += 1
            demanda_perdida[fecha] = demanda_fecha - inventario
            ventas += (inventario)
            inventario = 0
        else:
            ventas += demanda_fecha
            inventario -= demanda_fecha

        if contador_dias_pasados % T == 0:
            if inventario < r and orden_en_camino is False:
                ordenes_realizadas += 1
                cantidad_comprar = Q
                lista_fechas_compras.append(fecha)
                compras[fecha + timedelta(days=leadtime)] = cantidad_comprar
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

    demanda_perdida = sum(demanda_perdida.values())

    return (ganancias, lista_fechas, list(inventario_fechas.values())[1:], compras, lista_fechas_compras,
            ventas, ordenes_realizadas, quiebres_stock, demanda_perdida, cantidad_comprada, inventario_fechas)
