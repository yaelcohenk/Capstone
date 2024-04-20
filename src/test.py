import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm

from sklearn.ensemble import IsolationForest

datos = pd.read_excel(os.path.join("datos", "ventas_diarias_productos.xlsx"))
datos = datos[datos["Descripción"] == "pro plan alimento seco para adulto razas medianas 15 kg"]

isolation = IsolationForest()
datos["ISO"] = isolation.fit_predict(datos["Cantidad"].values.reshape(-1, 1))

print(f"El shape de datos antes de quitar outliers es {datos.shape}")

sns.lineplot(data=datos, x="Fecha", y="Cantidad")
plt.show()

decomposition = sm.tsa.seasonal_decompose(datos["Cantidad"], period=7)
fig = decomposition.plot()
fig.set_size_inches((16, 9))
fig.suptitle(f'Información ventas diarias del producto', fontsize=20)
fig.tight_layout()
plt.show()


datos = datos[datos["ISO"] != -1]
print(f"El shape de datos luego de quitar outliers es {datos.shape}")


decomposition = sm.tsa.seasonal_decompose(datos["Cantidad"], period=7)
fig = decomposition.plot()
fig.set_size_inches((16, 9))
fig.suptitle(f'Información ventas diarias del producto', fontsize=20)
fig.tight_layout()
plt.show()