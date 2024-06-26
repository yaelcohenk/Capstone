def calcular_metricas(resultados, lista_fechas, porcentaje=0, leadtime_on=True):
    if leadtime_on:
        print(f"Para el aumento en {porcentaje}% leadtime las métricas son")
    utilidad_total = 0
    ventas_totales = 0
    ordenes_realizadas_total = 0
    quiebres_stock_total = 0
    demanda_perdida_total = 0
    cantidad_comprada_total = 0
    costo_alm_total = 0
    costo_productos_vendidos = 0
    inventario_total = 0
    inventario_prom = 0
    
    inventarios_productos = dict()

    for elementos in resultados:
        costo = 0
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
        i = 0
        cantidad=0
        for fecha in lista_fechas:
            cantidad += int(inventario_fechas[fecha])
            i+=1
    
        inventario_prom += cantidad/i    
        
    
    nivel_rotacion = costo_productos_vendidos / inventario_prom


    print(f"La empresa dentro de todo el período registró utilidad por {utilidad_total} CLP")
    print(f"Se vendieron un total de {ventas_totales} productos")
    print(f"Se emitieron {ordenes_realizadas_total} órdenes de compra")
    print(f"Hubo quiebres de stock en {quiebres_stock_total} veces")
    print(f"La demanda perdida total alcanza el valor de {demanda_perdida_total} unidades")
    print(f"En total se compraron {cantidad_comprada_total} productos")
    print(f"El costo de almacenaje total fue de {costo_alm_total}")
    print(f"El nivel de rotación es de {nivel_rotacion}\n")

    return utilidad_total, ventas_totales, ordenes_realizadas_total, quiebres_stock_total, demanda_perdida_total, cantidad_comprada_total, costo_alm_total, nivel_rotacion