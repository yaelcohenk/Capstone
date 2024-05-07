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

# print(datos_forecasting)


# sys.exit()
datos_validacion = datos_forecasting[datos_forecasting["ds"].dt.year >= 2023]
datos_forecasting = datos_forecasting[datos_forecasting["ds"].dt.year < 2023]


# print(datos_validacion)
# print(datos_forecasting)

model = prophet.Prophet()
model.fit(datos_forecasting)

# future = model.make_future_dataframe(periods=30)
# print(future.tail())
# print(datos_validacion.columns)
# print(datos_validacion["ds"])
fechas_futuras = datos_validacion["ds"].to_frame()
print(fechas_futuras)

# print(fechas_futuras)
forecast = model.predict(fechas_futuras)

# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
# 
fig = model.plot(forecast)
plt.show()
