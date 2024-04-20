import prophet
import pandas as pd
import os
import matplotlib.pyplot as plt


ventas_prod = pd.read_excel(os.path.join("datos", "ventas_diarias_productos_vigentes_no_outliers.xlsx")) 
testeo = ventas_prod[ventas_prod["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]
datos_forecasting = testeo[["Fecha", "Cantidad"]]

datos_forecasting.columns = ["ds", "y"]


# print(datos_forecasting)
# datos_forecasting["z_score"] = (datos_forecasting["y"] - datos_forecasting["y"].mean()) / datos_forecasting["y"].std()

# print(datos_forecasting)

# datos_forecasting = datos_forecasting[(datos_forecasting["z_score"] < 3) & (datos_forecasting["z_score"] > -3)]
# print(datos_forecasting)
model = prophet.Prophet()
model.fit(datos_forecasting)

future = model.make_future_dataframe(periods=30)
print(future.tail())
forecast = model.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig = model.plot(forecast)
plt.show()
plt.close()

fig2 = model.plot_components(forecast)
plt.show()