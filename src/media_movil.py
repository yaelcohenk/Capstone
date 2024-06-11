import pandas as pd
import sys
import os
import json
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error,
                             mean_absolute_error,
                             mean_squared_error)
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
from datetime import date
import prophet


ventas_prod = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES) 
testeo = ventas_prod[ventas_prod["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]
datos_forecasting = testeo[["Fecha", "Cantidad"]]
datos_forecasting['year_month']=datos_forecasting['Fecha'].apply(lambda x:pd.Timestamp(x).strftime('%Y-%m'))
mensual= datos_forecasting.groupby(by=['year_month'])['Cantidad'].sum().reset_index()
mensual["year_month"] = pd.to_datetime(mensual["year_month"])
mensual.columns = ["ds", "y"]



datos_enteros=mensual
datos_validacion = mensual[mensual["ds"].dt.year >= 2023 ]
datos_forecasting = mensual[mensual["ds"].dt.year < 2023]

#model = ARIMA(datos_forecasting['y'], order=(0, 0, 12))  # El orden es (p, d, q), aquí estamos usando un modelo MA(2)
#model_fit = model.fit()

#model = prophet.Prophet()
#model.fit(datos_forecasting)


fechas_futuras = datos_validacion["ds"].to_frame()


# print(fechas_futuras)
#forecast = model.predict(start=fechas_futuras)
#predicciones = forecast["yhat"]
#valores_reales = datos_validacion["y"]

#model = prophet.Prophet()
#model.fit(datos_forecasting)




datos_enteros["yhat"]=datos_enteros["y"].rolling(window=6).mean().shift(6)
test=datos_enteros[36:]
largo=len(datos_forecasting)
print(largo)
#forecast = model_fit.predict(fechas_futuras)

predicciones = test["yhat"]
valores_reales = test["y"]

lista_fechas = fechas_futuras["ds"].values
predicciones = predicciones.values.tolist()
valores_reales = valores_reales.values.tolist()

predicciones = [int(i) for i in predicciones]
errores=[]
suma=0
for i in len(valores_reales):
    error=valores_reales[i]-predicciones[i]
    suma+=error
    errores.append(error)
MAD_prophet=np.mean(np.abs(errores))
if MAD_prophet!=0:
    tracking_signal=suma/MAD_prophet
else:
    tracking_signal = np.nan
print(lista_fechas)
print(predicciones)
print(valores_reales)


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


plt.plot(datos["predicciones"], color="red", label="Predicción")
plt.plot(datos["real"], color="blue", label="Valor Real")
plt.title("Predicciones Media Movil en set de datos validación")
plt.legend()
plt.show()