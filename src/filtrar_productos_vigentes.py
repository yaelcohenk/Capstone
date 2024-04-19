import pandas as pd
import os

ventas_diarias = pd.read_excel(os.path.join("datos", "ventas_diarias_productos.xlsx"))
fecha_venta_max = ventas_diarias.groupby("Descripción").agg({"Fecha": "max"}).reset_index()
filtro_objetos_venta = fecha_venta_max[fecha_venta_max["Fecha"] >= "2023-12-1"]
filtro_objetos_venta = filtro_objetos_venta.drop("Fecha", axis=1)
filtro_objetos_venta.to_excel(os.path.join("datos", "prod_vigentes.xlsx"), index=False)


