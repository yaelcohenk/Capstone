import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


def graficar_predicciones_pronostico(ruta_archivos):
    archivos = os.listdir(ruta_archivos)

    dataframes = list()

    for archivo in archivos:
        if archivo.endswith(".txt"):
            with open(os.path.join(ruta_archivos, archivo), "r") as f:
                mapeos = json.load(f)

    for archivo in archivos:
        if archivo.endswith(".xlsx"):
            numero = archivo.split(".")[0].split("_")[1]
            dataframe = pd.read_excel(os.path.join(ruta_archivos, archivo))
            dataframe.columns = ["Fechas", "Real", "Predicción"]
            dataframes.append((dataframe, mapeos[numero]))

    return dataframes


nombre_art = "kong extreme dental con cuerda"
archivos_xgboost = graficar_predicciones_pronostico(
    os.path.join("predicciones", "xgboost"))
archivos_prophet = graficar_predicciones_pronostico(
    os.path.join("predicciones", "prophet"))
archivos_holt = graficar_predicciones_pronostico(
    os.path.join("predicciones", "holt_winters"))

archivos_producto = list()

for archivo, nombre in archivos_xgboost:
    if nombre == nombre_art:
        archivos_producto.append(archivo)

for archivo, nombre in archivos_prophet:
    if nombre == nombre_art:
        archivo.columns = ["Fecha", "Predicción", "Real"]
        archivos_producto.append(archivo)

for archivo, nombre in archivos_holt:
    if nombre == nombre_art:
        archivo.columns = ["Fecha", "Predicción", "Real"]
        archivos_producto.append(archivo)


datos = pd.DataFrame({"Fechas": archivos_producto[1]["Fecha"],
         "XGBoost": archivos_producto[0]["Predicción"],
         "Prophet": archivos_producto[1]["Predicción"],
         "Holt Winters": archivos_producto[2]["Predicción"],
         "Real": archivos_producto[0]["Real"]})


datos = datos.head(30)

plt.figure(figsize=(25, 6))
datos['Real'].plot(style='b', figsize=(10, 5), label='Original', linewidth=2.5)
datos['XGBoost'].plot(style='r', figsize=(10, 5), label='Predicción XGBoost', linewidth=2)
datos['Prophet'].plot(style='g', figsize=(10, 5), label='Predicción Prophet', linewidth=1.5)
datos['Holt Winters'].plot(style='y', figsize=(10, 5), label='Predicción Holt Winters')

plt.xlabel('Fecha')
plt.ylabel('Cantidad')
plt.title(f'Predicciones {nombre_art} para los distintos modelos')
plt.legend()
plt.show()


# print(archivos_producto)
# print(archivos_xgboost)

# pass
# archivos = os.listdir(os.path.join("predicciones", "xgboost"))
#
# dataframes = list()
# for archivo in archivos:
# if archivo.endswith(".txt"):
# with open(os.path.join("predicciones", "xgboost", archivo), "r") as f:
# mapeos = json.load(f)
#
#
# for archivo in archivos:
# if archivo.endswith(".xlsx"):
# numero = archivo.split(".")[0].split("_")[1]
# dataframe = pd.read_excel(os.path.join("predicciones", "xgboost", archivo))
# dataframes.append((dataframe, mapeos[numero]))
#
# contador = 0
# for dataframe, nombre in dataframes:
# print(dataframe.columns)
# dataframe.columns = ["tiempo", "predicciones", "real"]
# dataframe.set_index("tiempo", inplace=True)
# plt.figure(figsize=(25, 6))
# dataframe['real'].plot(style='b', figsize=(10, 5), label='Original')
# dataframe['predicciones'].plot(style='r', figsize=(10, 5), label='Predicción')

# plt.xlabel('Fecha')
# plt.ylabel('Cantidad')
# plt.title(f'Predicciones XGBoost para {nombre}')
# plt.legend()
# plt.savefig(os.path.join("plots", "forecasts", "xgboost", f"prod_{contador}.png"), dpi=600)
# plt.close()
# print(f"Voy en la iteración {contador}")
# contador += 1
