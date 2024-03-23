import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging


from parametros import PATH_SALES_DATA, PATH_VENTAS_ITEM_IDS

logging.basicConfig(level=logging.DEBUG)
plt.set_loglevel(level = 'warning')

datos_venta = pd.read_csv(PATH_SALES_DATA, sep=";")
datos_venta["date"] = pd.to_datetime(datos_venta.date, format='%Y-%m-%d')

logging.debug("Los datos de venta son\n")
logging.debug(datos_venta.head())

logging.debug(f"Los valores que faltan son:\n {datos_venta.isnull().sum()} \n")
logging.debug(f"Los datos de venta son\n {datos_venta.info()}")

logging.debug("La descripción estadística de las columnas numéricas son \n")
logging.debug(datos_venta.describe(include=[np.number]).T)

# Con esto sacamos la cantidad de veces que se ha vendido un ítem
ventas_item = datos_venta.item_id.value_counts().reset_index()
ventas_item.columns = ["item_id", "cantidad_vendida"]

fig, ax = plt.subplots()
lineplot = sns.lineplot(data=ventas_item, x="item_id", y="cantidad_vendida")
plt.xlabel("Item ID")
plt.ylabel("Cantidad vendida")
plt.title("Cantidad vendida de un cierto ítem de manera histórica")

fig = lineplot.get_figure()
fig.savefig(PATH_VENTAS_ITEM_IDS)

plt.close()