import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from funciones.adi_categorizacion import category

# pd.options.mode.chained_assignment = None  # default='warn'
productos_vigentes = pd.read_excel(os.path.join("datos", "prod_vigentes.xlsx"))
productos_vigentes = productos_vigentes["Descripción"].to_list()

ventas_diarias_prod = pd.read_excel(os.path.join("datos", "ventas_diarias_productos.xlsx"))
ventas_diarias_prod = ventas_diarias_prod[ventas_diarias_prod["Descripción"].isin(productos_vigentes)]

productos_unicos = ventas_diarias_prod["Descripción"].unique().tolist()

listas = list()

for producto in productos_unicos:
    producto_db = ventas_diarias_prod[ventas_diarias_prod["Descripción"].isin([producto])]
    producto_db = producto_db.sort_values(by="Fecha")
    producto_db["diferencia_tiempo"] = producto_db["Fecha"].diff()
    adi = producto_db["diferencia_tiempo"].mean().days
    ventas_totales = producto_db["Cantidad"].sum()
    cv_sqr = (producto_db["Cantidad"].std() / producto_db["Cantidad"].mean()) ** 2  
    listas.append([producto, adi, cv_sqr, ventas_totales])


dataset = pd.DataFrame(listas, columns=["Descripción", "ADI", "CV2", "Ventas_totales"])

dataset["categoria"] = dataset.apply(category, axis=1)
# sns.scatterplot(x='CV2',y='ADI',hue='categoria',data=dataset)
# plt.show()

print(dataset.categoria.value_counts())
dataset.to_excel(os.path.join("datos", "adi_productos.xlsx"), index=False)
