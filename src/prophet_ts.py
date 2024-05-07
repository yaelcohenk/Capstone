import prophet
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES


ventas_prod = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES) 
testeo = ventas_prod[ventas_prod["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]
datos_forecasting = testeo[["Fecha", "Cantidad"]]
datos_forecasting["Fecha"] = pd.to_datetime(datos_forecasting["Fecha"])
datos_forecasting.columns = ["ds", "y"]


datos_validacion = datos_forecasting[datos_forecasting["ds"].dt.year >= 2023]
datos_forecasting = datos_forecasting[datos_forecasting["ds"].dt.year < 2023]


model = prophet.Prophet()
model.fit(datos_forecasting)


fechas_futuras = datos_validacion["ds"].to_frame()


# print(fechas_futuras)
forecast = model.predict(fechas_futuras)
predicciones = forecast["yhat"]
valores_reales = datos_validacion["y"]

lista_fechas = fechas_futuras["ds"].values
predicciones = predicciones.values.tolist()
valores_reales = valores_reales.values.tolist()

predicciones = [int(i) for i in predicciones]

print(lista_fechas)
print(predicciones)
print(valores_reales)

datos = pd.DataFrame({"predicciones": predicciones, "real": valores_reales})
datos.set_index(lista_fechas, inplace=True)

print(datos)

plt.plot(datos["predicciones"], color="red", label="Predicción")
plt.plot(datos["real"], color="blue", label="Valor Real")
plt.title("Predicciones Prophet en set de datos validación")
plt.legend()
plt.show()


# print(predicciones, valores_reales)

# print(type(predicciones), predicciones.shape)


# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
# 
# fig = model.plot(forecast)
# plt.show()
