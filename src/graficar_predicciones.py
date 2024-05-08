import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def graficar_predicciones_pronostico():
    pass
archivos = os.listdir(os.path.join("predicciones", "prophet"))

dataframes = list()
for archivo in archivos:
    if archivo.endswith(".txt"):
        with open(os.path.join("predicciones", "prophet", archivo), "r") as f: 
            mapeos = json.load(f)


for archivo in archivos:
    if archivo.endswith(".xlsx"):
        numero = archivo.split(".")[0].split("_")[1]
        dataframe = pd.read_excel(os.path.join("predicciones", "prophet", archivo)) 
        dataframes.append((dataframe, mapeos[numero]))

contador = 0
for dataframe, nombre in dataframes:
    # dataframe.columns = ["tiempo", "predicciones", "real"]
    # dataframe.set_index("tiempo", inplace=True)
    # plt.figure(figsize=(25, 6))
    # dataframe['real'].plot(style='b', figsize=(10, 5), label='Original')
    # dataframe['predicciones'].plot(style='r', figsize=(10, 5), label='Predicción')
# 
    # plt.xlabel('Fecha')
    # plt.ylabel('Cantidad')
    # plt.title(f'Predicciones Holt Winters para {nombre}')
    # plt.legend()
    # plt.savefig(os.path.join("plots", "forecasts", "holt_winters", f"prod_{contador}.png"), dpi=600)
    # plt.close()
    # print(f"Voy en la iteración {contador}")
    # contador += 1
