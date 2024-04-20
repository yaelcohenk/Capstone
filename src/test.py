import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm

from sklearn.ensemble import IsolationForest
pd.options.mode.chained_assignment = None 

productos_vigentes = pd.read_excel(os.path.join("datos", "prod_vigentes.xlsx"))
productos_vigentes = productos_vigentes["Descripción"].tolist()

datos = pd.read_excel(os.path.join("datos", "ventas_diarias_productos.xlsx"))

datos = datos[datos["Descripción"].isin(productos_vigentes)]


lista_dataframes = list()

print(f"El tamaño de los datos antes de quitar outliers es {datos.shape}")

for producto in productos_vigentes:

    producto_data = datos[datos["Descripción"].isin([producto])]
    isolation = IsolationForest()
    producto_data["ISO"] = isolation.fit_predict(producto_data["Cantidad"].values.reshape(-1, 1))
    producto_data = producto_data[producto_data["ISO"] != -1]

    lista_dataframes.append(producto_data)


datos_final = pd.concat(lista_dataframes)

print(f" El tamaño de los datos luego de quitar outliers es {datos_final.shape}")

datos_final.to_excel(os.path.join("datos", "ventas_diarias_productos_vigentes_no_outliers.xlsx"), index=False)










# print(datos)
# print(len(datos["Descripción"].unique()))

# datos = datos[datos["Descripción"] == "pro plan alimento seco para adulto razas medianas 15 kg"]



# isolation = IsolationForest()
# datos["ISO"] = isolation.fit_predict(datos["Cantidad"].values.reshape(-1, 1))
# 
# print(f"El shape de datos antes de quitar outliers es {datos.shape}")
# 
# sns.lineplot(data=datos, x="Fecha", y="Cantidad")
# plt.show()
# 
# decomposition = sm.tsa.seasonal_decompose(datos["Cantidad"], period=7)
# fig = decomposition.plot()
# fig.set_size_inches((16, 9))
# fig.suptitle(f'Información ventas diarias del producto', fontsize=20)
# fig.tight_layout()
# plt.show()
# 
# 
# datos = datos[datos["ISO"] != -1]
# print(f"El shape de datos luego de quitar outliers es {datos.shape}")
# 
# 
# decomposition = sm.tsa.seasonal_decompose(datos["Cantidad"], period=7)
# fig = decomposition.plot()
# fig.set_size_inches((16, 9))
# fig.suptitle(f'Información ventas diarias del producto', fontsize=20)
# fig.tight_layout()
# plt.show()