import pandas as pd
import os
import sys


from funciones.features import (create_features,
                                apply_trend_and_seasonal,
                                apply_fourier_transform,
                                apply_day_selling_differences)


ventas_prod = pd.read_excel(os.path.join("datos", "ventas_diarias_productos_vigentes_no_outliers.xlsx"))

ventas_prod.drop("ISO", axis=1, inplace=True)
productos_vigentes = ventas_prod["Descripción"].unique().tolist()
ventas_prod.index = ventas_prod["Fecha"]

lista_dataframes = list()

for producto in productos_vigentes:
    producto_ventas = ventas_prod[ventas_prod["Descripción"].isin([producto])]

    datos_forecast = apply_day_selling_differences(producto_ventas)
    datos_forecast = apply_fourier_transform(datos_forecast)
    datos_forecast = apply_trend_and_seasonal(datos_forecast)
    datos_forecast = create_features(datos_forecast)

    # Poner demanda acumulada quizás de los últimos x días, demanda de ayer, demanda de los últimos
    # días
    lista_dataframes.append(datos_forecast)

datos_final = pd.concat(lista_dataframes)
datos_final.to_excel(os.path.join("datos", "ventas_diarias_prod_vigentes_no_outliers_w_features.xlsx"))
