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
modelos_holt_winters = [producto for producto, pronostico in datos_modelos if pronostico == "holt_winters"]

mapeos_croston = os.path.join("predicciones", "croston", "mapeos.txt")
mapeos_prophet = os.path.join("predicciones", "prophet", "mapeos.txt")
mapeos_xgboost = os.path.join("predicciones", "xgboost", "mapeos.txt")
mapeos_holt_winters = os.path.join("predicciones", "holt_winters", "mapeos.txt")

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
mapeos_holt_winters = rutas_modelos(modelos_holt_winters, mapeos_holt_winters)

# datos = pd.read_excel(os.path.join("predicciones", "prophet", "producto_151.xlsx"))
# datos["error"] = datos["real"] - datos["predicciones"]
# 
# xmin = np.min(datos["error"])
# xmax = np.max(datos["error"])
# mu, std = norm.fit(datos["error"])


distribuciones = dict()

def distribuciones_prod(mapeos, modelo, data_real, data_pred, graficar_errores=False):
    distribuciones = dict()
    # errores_productos = dict()
    contador = 0
    for key, value in mapeos.items():
        datos = pd.read_excel(os.path.join("predicciones", modelo, f"producto_{value}.xlsx"))
        datos["error"] = datos[data_real] - datos[data_pred]


        xmin = np.min(datos["error"])
        xmax = np.max(datos["error"])
        mu, std = norm.fit(datos["error"])

        distribuciones[key] = (mu, std)
        if graficar_errores:
            plt.figure(figsize=(10, 8))
            sns.histplot(x=datos["error"], bins=30, kde=True)
            # sns.displot(x=datos["error"], kde=True)
            # sns.barplot(x=datos["error"])
            plt.title(f"Histograma errores para {key}")
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            # valores_normal = np.random.normal(mu, std, 1000)
            # plt.plot(valores_normal, marker="o", linestyle="-")
            # plt.plot(x, p, 'k', linewidth=2)
            plt.savefig(os.path.join("errores", f"grafico_{contador}.png"))
            contador += 1
            plt.close()


        # errores_productos[key] = datos["error"]

    return distribuciones

distr_proph = distribuciones_prod(mapeos_prophet, "prophet", "real", "predicciones")
distr_xgboost = distribuciones_prod(mapeos_xgboost, "xgboost", "valor_real", "valor_prediccion", graficar_errores=True)
distr_croston = distribuciones_prod(mapeos_croston, "croston", "real", "predicciones")
distr_holt_winters = distribuciones_prod(mapeos_holt_winters, "holt_winters", "real", "predicciones")

distrs = dict()
distrs.update(distr_proph)
distrs.update(distr_xgboost)
distrs.update(distr_croston)
distrs.update(distr_holt_winters)





# print(len(distrs))

# with open("distrs.pkl", "wb") as f:
        # pickle.dump(distrs, f)

