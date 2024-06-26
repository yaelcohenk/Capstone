import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from scipy.stats import norm

datos_modelos = pd.read_excel(os.path.join("datos", "Selección_modelo.xlsx"))
datos_modelos = datos_modelos[["Descripción", "Modelo de pronostico"]]

datos_modelos = datos_modelos.to_numpy()
modelos_croston = [producto for producto, pronostico in datos_modelos if pronostico == "croston"]
modelos_prophet = [producto for producto, pronostico in datos_modelos if pronostico == "prophet"]
modelos_xgboost = [producto for producto, pronostico in datos_modelos if pronostico == "xgboost"]


mapeos_croston = os.path.join("predicciones", "croston", "mapeos.txt")
mapeos_prophet = os.path.join("predicciones", "prophet", "mapeos.txt")
mapeos_xgboost = os.path.join("predicciones", "xgboost", "mapeos.txt")


def rutas_modelos(modelos, ruta):

    with open(ruta, "r") as f:
        diccionario = json.load(f)
        nuevo_diccionario = dict()
        for key, value in diccionario.items():
            if value in modelos:
                nuevo_diccionario[value] = key

    return nuevo_diccionario


mapeos_xgboost = rutas_modelos(modelos_xgboost, mapeos_xgboost)
mapeos_croston = rutas_modelos(modelos_croston, mapeos_croston)
mapeos_prophet = rutas_modelos(modelos_prophet, mapeos_prophet)


datos = pd.read_excel(os.path.join("predicciones", "prophet", "producto_151.xlsx"))
datos["error"] = datos["real"] - datos["predicciones"]

xmin = np.min(datos["error"])
xmax = np.max(datos["error"])
mu, std = norm.fit(datos["error"])


distribuciones = dict()

def distribuciones_prod(mapeos, modelo, data_real, data_pred):
    distribuciones = dict()
    for key, value in mapeos.items():
        datos = pd.read_excel(os.path.join("predicciones", modelo, f"producto_{value}.xlsx"))
        datos["error"] = datos[data_real] - datos[data_pred]

        xmin = np.min(datos["error"])
        xmax = np.max(datos["error"])
        mu, std = norm.fit(datos["error"])

        distribuciones[key] = (mu, std)

    return distribuciones

distr_proph = distribuciones_prod(mapeos_prophet, "prophet", "real", "predicciones")
distr_xgboost = distribuciones_prod(mapeos_xgboost, "xgboost", "valor_real", "valor_prediccion")
distr_croston = distribuciones_prod(mapeos_croston, "croston", "real", "predicciones")

distrs = dict()
distrs.update(distr_proph)
distrs.update(distr_xgboost)
distrs.update(distr_croston)


with open("distrs.pkl", "wb") as f:
        pickle.dump(distrs, f)

