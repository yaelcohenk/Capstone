import pandas as pd
import os

ventas_diarias = pd.read_excel(os.path.join("datos", "ventas_diarias_productos.xlsx"))
fecha_venta_max = ventas_diarias.groupby("Descripción").agg({"Fecha": "max"}).reset_index()
filtro_objetos_venta = fecha_venta_max[fecha_venta_max["Fecha"] >= "2023-12-1"]
filtro_objetos_venta = filtro_objetos_venta.drop("Fecha", axis=1)

columnas = ["Descripción", "Cantidad"]
ventas = ventas_diarias.groupby("Descripción").agg({"Cantidad" : "sum"}).reset_index()
ventas_mayor_num = ventas.query("Cantidad > 30")

ventas_mayor_num_prod = ventas_mayor_num["Descripción"].tolist()

filtro_objetos_venta = filtro_objetos_venta[filtro_objetos_venta["Descripción"].isin(ventas_mayor_num_prod)]

filtro_objetos_venta.to_excel(os.path.join("datos", "prod_vigentes.xlsx"), index=False)

# Así como está, son 311 productos los que quedan finalmente