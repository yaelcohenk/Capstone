import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from parametros import PATH_STOCK_DATA, PATH_GRUPOS_INVENTARIO, PATH_SUBGRUPOS_INVENTARIO
from funciones.dataframes import dataframe_from_value_counts
from funciones.plotting import graficar_seaborn

logging.basicConfig(level=logging.DEBUG)
plt.set_loglevel(level='warning')


if __name__ == '__main__':
    datos_inventario = pd.read_csv(PATH_STOCK_DATA, sep=";")
    # Asumimos que hay un typo en esos grupos. Si no no tiene mucho sentido
    datos_inventario["group_description"].replace(
        "medicamento", "medicamentos", inplace=True)
    datos_inventario["group_description"].replace(
        "accesorio", "accesorios", inplace=True)

    logging.debug(datos_inventario.head())
    logging.debug(datos_inventario.isnull().sum())
    logging.debug(datos_inventario.info())
    logging.debug(datos_inventario.describe(include=[np.number]).T)

    grupos_inventario = dataframe_from_value_counts(
        datos_inventario.group_description, ["Grupos", "Cantidad"])

    graficar_seaborn(sns.barplot(grupos_inventario, y="Grupos", x="Cantidad"),
                     "Cantidad",
                     "Grupos",
                     "Cantidad en cada grupo",
                     PATH_GRUPOS_INVENTARIO)

    subgrupos_inventario = dataframe_from_value_counts(
        datos_inventario.description_2, ["Subgrupos", "Cantidad"])

    graficar_seaborn(sns.barplot(subgrupos_inventario, y="Subgrupos", x="Cantidad", orient="h"),
                     "Cantidad",
                     "Subgrupos",
                     "Cantidad items en cada subgrupo",
                     PATH_SUBGRUPOS_INVENTARIO)
