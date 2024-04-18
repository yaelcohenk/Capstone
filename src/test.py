import pandas as pd
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import sys

datos = pd.read_excel(os.path.join("datos", "ventas_diarias_productos.xlsx"))
# print(datos)
# sys.exit()
datos = datos[datos["Descripción"] == "pro plan alimento seco para adulto razas medianas 15 kg"]

cols = ["Fecha", "Cantidad"]
datos = datos[cols]

datos['Fecha'] = pd.to_numeric(pd.to_datetime(datos['Fecha']))

# print(datos)

plotted = plot_acf(datos["Cantidad"], lags=40)
plt.xlabel('Rezagos')
plt.ylabel('Autocorrelación')
plt.title('Función de Autocorrelación Muestral')
plotted.savefig("ejemplo_autocorrelacion.png")
plt.show()

