import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from parametros import PATH_STOCK_DATA, PATH_SALES_DATA, PATH_VENTAS_ANUALES_GRUPO, PATH_SUBGRUPOS_MAS_VENDIDOS
from funciones.plotting import graficar_seaborn, concatenar

logging.basicConfig(level=logging.DEBUG)
plt.set_loglevel(level='warning')


datos_inventario = pd.read_csv(PATH_STOCK_DATA, sep=";")
datos_inventario["group_description"].replace("medicamento", "medicamentos", inplace=True)
datos_inventario["group_description"].replace("accesorio", "accesorios", inplace=True)

datos_venta = pd.read_csv(PATH_SALES_DATA, sep=";")
datos_venta["date"] = pd.to_datetime(datos_venta.date, format='%Y-%m-%d')

# Aquí juntamos ambos dataframes, ya que quiero tener las características de los ítems para lo que compró cada cliente
datos_conjuntos = pd.merge(datos_venta, datos_inventario, on="item_id")

datos_conjuntos.to_excel(os.path.join("datos", "datos_conjuntos.xlsx"), index=False)

columnas_mantener = ["date", "quantity", "description_2",
                     "group_description", "total (CLP)", "description"]

database = datos_conjuntos[columnas_mantener]

logging.debug(database.info())

database["ano"] = database["date"].dt.year
database["mes"] = database["date"].dt.month

db_grupos_ano = database.groupby(["ano", "group_description"]).size().reset_index(name='count')


grafico = sns.barplot(data=db_grupos_ano, x="ano",
                      y="count", hue="group_description")
plt.xlabel("Año")
plt.ylabel("Tamaño ventas del grupo")
plt.title("Tamaño ventas de cada grupo por año")


graficar_seaborn(grafico,
                 "Año",
                 "Tamaño del grupo",
                 "Tamaño ventas de cada grupo por año",
                 PATH_VENTAS_ANUALES_GRUPO)


database_subgrupos_anual = database.groupby(
    ["ano", "description_2"]).size().reset_index(name='count')
df_grouped_sorted = database_subgrupos_anual.sort_values(
    by=['ano', 'count'], ascending=[True, False])


top_items_per_year = []
# Iterar sobre cada año y seleccionar los 5 elementos más vendidos
for year in df_grouped_sorted['ano'].unique():
    top_items_per_year.append(
        df_grouped_sorted[df_grouped_sorted['ano'] == year].head(7))

database_top_items = concatenar(top_items_per_year)

grafico = sns.barplot(data=database_top_items, x="ano",
                      y="count", hue="description_2")
plt.title('Top 7 de elementos más vendidos por año')
plt.xlabel('Año')
plt.ylabel('Cantidad vendida')

graficar_seaborn(grafico,
                 path=PATH_SUBGRUPOS_MAS_VENDIDOS,
                 size_x=10, size_y=6)


datos_agrupados = database.groupby(["ano", "mes", "group_description"]).size()
datos_agrupados = datos_agrupados.reset_index(name='count')

lista_anos = datos_agrupados["ano"].unique()
for ano in lista_anos:
    database_ano = datos_agrupados[datos_agrupados["ano"] == ano]
    fig, ax = plt.subplots(figsize=(15, 7))
    grafico = sns.barplot(x=database_ano["mes"], y=database_ano["count"], hue=database_ano["group_description"], ax=ax)
    plt.xlabel("Mes")
    plt.ylabel("Ventas")
    plt.title("Ventas por mes en cada año de los tres grandes grupos")
    ax.legend(title=f"Año {ano}", bbox_to_anchor=(1, 1), loc="upper left")
    
    path = os.path.join("plots", "eda_conjunto", f"ventas_grupos_ano_{ano}")
    fig = grafico.get_figure()
    fig.savefig(path)
    
    plt.close()
    # plt.show()