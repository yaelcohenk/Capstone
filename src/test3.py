import pandas as pd
import os
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES


datos_vigentes = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
ganancias = pd.read_excel(os.path.join("datos", "ganancias_total_productos.xlsx"))
productos = datos_vigentes["Descripción"].unique().tolist()

ganancias = ganancias[ganancias["description"].isin(productos)]

print(ganancias)

# Con el filtro de utilizar la venta de los productos luego de cierta fecha y que
# históricamente han tenido más de 30 ventas, en términos de ganancias históricas, se ven
# representadas el 99.92 % de las ganancias. Al dejar un total de casi 1300 prouctos de lado
# tan solo estamos obviando un 0.08% de las ganancias.