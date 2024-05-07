import pandas as pd

def parametros_producto(datos_producto: pd.DataFrame):
    precio_venta = datos_producto["unit_sale_price (CLP)"].to_numpy()[0]
    costo_compra = datos_producto["cost (CLP)"].to_numpy()[0]
    costo_almacenar = datos_producto["storage_cost (CLP)"].to_numpy()[0]
    leadtime = int(datos_producto["leadtime (days)"].to_numpy()[0])
    volumen = datos_producto["size_m3"].to_numpy()[0]
    costo_fijo_comprar = datos_producto["cost_per_purchase"].to_numpy()[0]
    item_id = datos_producto["item_id"].to_numpy()[0]
    nombre_prod = datos_producto["description"].to_numpy()[0]

    ids_asociados = datos_producto["item_id"].tolist()
    return precio_venta, costo_compra, costo_almacenar, leadtime, volumen, costo_fijo_comprar, item_id, nombre_prod