import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
datos = pd.read_excel(os.path.join("datos", "ventas_diarias_subgrupos.xlsx"))
datos2 = pd.read_excel(os.path.join("datos", "ventas_diarias_productos.xlsx"))

datos_ganancias = pd.read_excel(os.path.join("datos", "ganancias_total_productos.xlsx"))
datos_ganancias = datos_ganancias.head(75)
productos = datos_ganancias["description"]

ventas_75_prod_mas_vendidos = datos2[datos2["Descripción"].isin(productos)]


datos = ventas_75_prod_mas_vendidos.sort_values(by=["Fecha"], ascending=True)
datos = datos.set_index("Fecha")

datos = datos.sort_values(by=["Grupo"])

tabla_dinamica = datos.pivot_table(index='Fecha', columns='Descripción', values='Cantidad', aggfunc='sum')



# print(len(datos["Descripción"]))
datos_no_repetidos = datos[["Descripción", "Grupo"]].drop_duplicates()

nombres_orden_grupo = datos_no_repetidos["Descripción"]

tabla_dinamica = tabla_dinamica[nombres_orden_grupo]


fig, ax = plt.subplots(figsize=(25, 10))
sns.heatmap(tabla_dinamica.corr(), cmap="crest")
plt.title("Matriz correlación ventas diarias en los 75 productos más vendidos")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

plt.tight_layout()
fig.savefig("correlacion_ventas_diarias.png")
plt.show()
