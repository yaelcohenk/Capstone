import pandas as pd
import sys
import os
import json
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error,
                             mean_absolute_error,
                             mean_squared_error)
import numpy as np
import matplotlib.pyplot as plt
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
from datetime import date
from darts import TimeSeries
import prophet
from darts.models import Croston



ventas_prod = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES) 
testeo = ventas_prod[ventas_prod["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]
datos_forecasting = testeo[["Fecha", "Cantidad"]]
datos_forecasting["Fecha"] = pd.to_datetime(datos_forecasting["Fecha"])
datos_forecasting.columns = ["ds", "y"]
datos_validacion_demanda = datos_forecasting[datos_forecasting["ds"].dt.year >= 2023]
datos_forecasting = datos_forecasting.set_index('ds').asfreq('D', fill_value=0).reset_index()
datos_validacion = datos_forecasting[datos_forecasting["ds"].dt.year >= 2023]


datos_forecasting = datos_forecasting[datos_forecasting["ds"].dt.year < 2023]
series_train = TimeSeries.from_dataframe(datos_forecasting, 'ds', 'y', fill_missing_dates=True, freq='D')
series_valid = TimeSeries.from_dataframe(datos_validacion, 'ds', 'y')



# Entrenar el modelo Croston
model = Croston(version="optimized")
model.fit(series_train)



fechas_futuras = datos_validacion_demanda["ds"].to_frame()



# print(fechas_futuras)
forecast = model.predict(len(fechas_futuras))
predicciones = forecast.values().flatten()
valores_reales = datos_validacion_demanda["y"]
lista_fechas = fechas_futuras["ds"].values
predicciones = predicciones.tolist()
valores_reales = valores_reales.values.tolist()


predicciones = [int(i) for i in predicciones]
errores=[]
suma=0
for i in range (len(valores_reales)):
    error=valores_reales[i]-predicciones[i]
    suma+=error
    errores.append(error)
MAD_prophet=np.mean(np.abs(errores))
if MAD_prophet!=0:
    tracking_signal=suma/MAD_prophet
else:
    tracking_signal = np.nan




datos = pd.DataFrame({"predicciones": predicciones, "real": valores_reales})
datos.set_index(lista_fechas, inplace=True)
mape = mean_absolute_percentage_error(valores_reales, predicciones)
rmse = root_mean_squared_error(valores_reales, predicciones)
mae = mean_absolute_error(valores_reales, predicciones)
mse = mean_squared_error(valores_reales, predicciones)
print(mape)
print(rmse)
print(mae)
print(mse)

print(datos)

plt.plot(datos["predicciones"], color="red", label="Predicción")
plt.plot(datos["real"], color="blue", label="Valor Real")
plt.title("Predicciones Croston en set de datos validación")
plt.legend()
plt.show()