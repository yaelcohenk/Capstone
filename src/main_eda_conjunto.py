import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import sys

from parametros import (PATH_STOCK_DATA,
                        PATH_SALES_DATA,
                        PATH_VENTAS_ANUALES_GRUPO,
                        PATH_SUBGRUPOS_MAS_VENDIDOS,
                        PATH_VENTAS_DIARIAS_GRUPO_DATA,
                        PATH_VENTAS_DIARIAS_SUBGRUPO_DATA,
                        PATH_VENTAS_DIARIAS_PRODUCTOS_DATA,
                        PATH_VENTAS_SEMANALES_GRUPO_DATA,
                        PATH_VENTAS_SEMANALES_SUBGRUPO_DATA,
                        PATH_VENTAS_SEMANALES_PRODUCTOS_DATA)

from funciones.plotting import graficar_seaborn, concatenar, graficar_ganancias

logging.basicConfig(level=logging.DEBUG)
plt.set_loglevel(level='warning')

graficar = False
datos_inventario = pd.read_csv(PATH_STOCK_DATA, sep=";")
datos_inventario["group_description"].replace(
    "medicamento", "medicamentos", inplace=True)
datos_inventario["group_description"].replace(
    "accesorio", "accesorios", inplace=True)

datos_venta = pd.read_csv(PATH_SALES_DATA, sep=";")
datos_venta["date"] = pd.to_datetime(datos_venta.date, format='%Y-%m-%d')

# Aquí juntamos ambos dataframes, ya que quiero tener las características de los ítems para lo que compró cada cliente
datos_conjuntos = pd.merge(datos_venta, datos_inventario, on="item_id")

datos_conjuntos.to_excel(os.path.join(
    "datos", "datos_conjuntos.xlsx"), index=False)

columnas_mantener = ["date", "quantity", "description_2",
                     "group_description", "total (CLP)", "description", "cost (CLP)"]

database = datos_conjuntos[columnas_mantener]

logging.debug(database.info())

database["ano"] = database["date"].dt.year
database["mes"] = database["date"].dt.month

db_grupos_ano = database.groupby(
    ["ano", "group_description"]).size().reset_index(name='count')


grafico = sns.barplot(data=db_grupos_ano, x="ano",
                      y="count", hue="group_description")
plt.xlabel("Año")
plt.ylabel("Tamaño ventas del grupo")
plt.title("Tamaño ventas de cada grupo por año")

if graficar:
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

if graficar:
    graficar_seaborn(grafico,
                     path=PATH_SUBGRUPOS_MAS_VENDIDOS,
                     size_x=10, size_y=6)


datos_agrupados = database.groupby(["ano", "mes", "group_description"]).size()
datos_agrupados = datos_agrupados.reset_index(name='count')

lista_anos = datos_agrupados["ano"].unique()
if graficar:
    for ano in lista_anos:
        database_ano = datos_agrupados[datos_agrupados["ano"] == ano]
        fig, ax = plt.subplots(figsize=(15, 7))
        grafico = sns.barplot(
            x=database_ano["mes"], y=database_ano["count"], hue=database_ano["group_description"], ax=ax)
        plt.xlabel("Mes")
        plt.ylabel("Ventas")
        plt.title("Ventas por mes en cada año de los tres grandes grupos")
        ax.legend(title=f"Año {ano}", bbox_to_anchor=(1, 1), loc="upper left")

        path = os.path.join("plots", "eda_conjunto",
                            f"ventas_grupos_ano_{ano}")
        fig = grafico.get_figure()
        fig.savefig(path)

        plt.close()


# Análisis de las ventas de cada uno de los tres grupos grandes de manera diaria
database_grupos_grandes_diario = database.groupby(
    ["group_description", "date"]).size().reset_index()
database_grupos_grandes_diario.columns = ["Grupo", "Fecha", "Ventas"]
database_grupos_grandes_diario.to_excel(
    PATH_VENTAS_DIARIAS_GRUPO_DATA, index=False)
# Aquí guardar el plot quizás, si es que lo necesitamos. Traspasarlo del jupyter
# Análisis ventas 49 subcategorías forma diaria
database_subcategorias_diario = database.groupby(
    ["description_2", "date"]).size().reset_index()
database_subcategorias_diario.columns = ["Subcategoría", "Fecha", "Cantidad"]
database_subcategorias_diario.to_excel(
    PATH_VENTAS_DIARIAS_SUBGRUPO_DATA, index=False)
# De ahí poner el código para guardar los plots. Traspasarlo del jupyter, dejarlo en una función
# o algo así
# Análisis ventas 1645 productos forma diaria
# database_productos_diario = database.groupby(["description", "date"]).size().reset_index()

database_productos_diario = database.groupby(["description", "date"]).agg({'quantity': 'sum', 'group_description': 'first'}).reset_index()

# print(resultado)
# print(database_productos_diario)
# print(database)
# sys.exit()
database_productos_diario.columns = [
    "Descripción", "Fecha", "Cantidad", "Grupo"]
database_productos_diario.to_excel(
    PATH_VENTAS_DIARIAS_PRODUCTOS_DATA, index=False)

# Ahora viene el análisis semanal
copy = database.rename(columns={"date": "fecha"})

copy["fecha"] = copy["fecha"] - pd.to_timedelta(7, unit="d")
# Lo de abajo simplemente sirve para juntar los datos por semana
# Primero análisis semanal de los 3 grandes grupos

database_grupos_grandes_semanal = (copy.groupby(['group_description', pd.Grouper(key='fecha', freq='W-MON')])['quantity']
                                   .size()
                                   .reset_index()
                                   .sort_values('fecha'))

database_grupos_grandes_semanal.to_excel(
    PATH_VENTAS_SEMANALES_GRUPO_DATA, index=False)

# Ventas semanales de los subgrupos
database_subcategorias_semanal = (copy.groupby(['description_2', pd.Grouper(key='fecha', freq='W-MON')])['quantity']
                                  .size()
                                  .reset_index()
                                  .sort_values('fecha'))

database_subcategorias_semanal.to_excel(
    PATH_VENTAS_SEMANALES_SUBGRUPO_DATA, index=False)

# Ventas semanalaes cada uno de los productos
database_productos_semanal = (copy.groupby(['description', pd.Grouper(key='fecha', freq='W-MON')])['quantity']
                              .size()
                              .reset_index()
                              .sort_values('fecha'))

database_productos_semanal.to_excel(
    PATH_VENTAS_SEMANALES_PRODUCTOS_DATA, index=False)

# Luego vemos las ganancias. De momento es histórico. De ahí verlo por mes y semana quizás

ganancias_total_subcategoria = database.groupby(
    ["description_2"]).sum().reset_index()
ganancias = ganancias_total_subcategoria["total (CLP)"].sum()
ganancias_total_subcategoria[
    "porcentaje_ganancias"] = ganancias_total_subcategoria["total (CLP)"] / ganancias

porcentajes = ganancias_total_subcategoria[[
    "description_2", "porcentaje_ganancias"]]


# Ordenar esto de ahí si es necesario
if graficar:
    graficar_ganancias(porcentajes,
                       "description_2",
                       "porcentaje_ganancias",
                       "Categoría",
                       "Porcentaje representativo ganancias",
                       'Porcentajes de las ganancias de cada uno de los subgrupos de manera histórica',
                       os.path.join("plots", "eda_conjunto", "ganancias_subgrupos.png"))


ganancias_total_grupos = database.groupby(
    ["group_description"]).sum().reset_index()
ganancias_grupos = ganancias_total_grupos["total (CLP)"].sum()
ganancias_total_grupos["porcentaje_ganancias"] = ganancias_total_grupos[
    "total (CLP)"] / ganancias_grupos

if graficar:
    graficar_ganancias(ganancias_total_grupos,
                       "group_description",
                       "porcentaje_ganancias",
                       "Categoría",
                       "Porcentaje representativo ganancias",
                       "Porcentajes de las ganancias de cada uno de los grupos de manera histórica",
                       os.path.join("plots", "eda_conjunto", "ganancias_grupos.png"))


# columnas_a_sumar = ["quantity", "total (CLP)"]

columnas_a_sumar = {"quantity": "sum",
                    "total (CLP)": "sum", "cost (CLP)": "first", "group_description": "first"}

logging.debug(database)
# ganancias_total_productos = database.groupby(["description"])[columnas_a_sumar].sum().reset_index()
ganancias_total_productos = database.groupby(
    ["description"]).agg(columnas_a_sumar).reset_index()
ganancias_productos = ganancias_total_productos["total (CLP)"].sum()
ganancias_total_productos["porcentaje_ganancias"] = ganancias_total_productos[
    "total (CLP)"] / ganancias_productos
ganancias_total_productos.sort_values(
    by=["porcentaje_ganancias"], ascending=False, inplace=True)

ganancias_total_productos["porcentaje_acumulado"] = ganancias_total_productos["porcentaje_ganancias"].cumsum()
ganancias_total_productos = ganancias_total_productos.reset_index(drop=True)

# Con esto se ve que hay 75 productos que en total representan el 80% de las ganancias históricas de la tienda de mascotas.
# Aprox el 4% de los productos ofrecidos por la tienda, representan el 80% de las ganancias históricas.


# logging.debug(database)
# logging.debug(ganancias_total_productos.head(75))
logging.debug(ganancias_total_productos)

ganancias_total_productos["ingreso_marginal"] = ganancias_total_productos[
    "total (CLP)"] - ganancias_total_productos["quantity"] * ganancias_total_productos["cost (CLP)"]

logging.debug(ganancias_total_productos)
# ganancias_total_productos.sort_values(by=["porcentaje_ganancias"], ascending=False, inplace=True)

ganancias_marginales_totales = ganancias_total_productos["ingreso_marginal"].sum(
)
ganancias_total_productos["porcentaje_ingreso_marginal"] = ganancias_total_productos["ingreso_marginal"] / \
    ganancias_marginales_totales

ganancias_total_productos.sort_values(
    by=["porcentaje_ingreso_marginal"], ascending=False, inplace=True)
ganancias_total_productos["porcentaje_marginal_acumulado"] = ganancias_total_productos["porcentaje_ganancias"].cumsum()


logging.debug(ganancias_total_productos)

ganancias_total_productos.to_excel(os.path.join(
    "datos", "ganancias_total_productos.xlsx"), index=False)
