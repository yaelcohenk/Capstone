import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sys
datos_escenarios_s_s = pd.read_excel("s_s_pronosticos_escenarios.xlsx")
datos_escenarios_s_s_tcero = pd.read_excel(
    "s_s_pronosticos_escenarios_tcero.xlsx")
datos_escenarios_r_q = pd.read_excel("r_q_pronosticos_escenarios.xlsx")
datos_escenarios_r_q_tcero = pd.read_excel(
    "r_q_pronosticos_escenarios_tcero.xlsx")


print(datos_escenarios_s_s)
print(datos_escenarios_s_s_tcero)


def graficar_histogramas_2(data_list, x, bins, label_list, title, xlabel, ylabel="Conteo", ruta=""):

    plt.figure(figsize=(10, 6))

    for indice, data in enumerate(data_list):
        sns.histplot(data=data, x=x, bins=bins, label=label_list[indice], alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(ruta)
    plt.close()


datos = [datos_escenarios_s_s,
         datos_escenarios_s_s_tcero,
         datos_escenarios_r_q,
         datos_escenarios_r_q_tcero]

labels = ["Histograma utilidad total (T, s, S)",
          "Histograma utilidad total (T, s, S) con T = 0",
          "Histograma utilidad total (T, r, Q)",
          "Histograma utilidad total (T, r, Q) con T = 0"]


graficar_histogramas_2(datos,
                       "utilidad_total",
                       30,
                       labels,
                       "Utilidad distintas políticas en distintos escenarios",
                       "Utilidad total (en decenas de millones de pesos)",
                       ruta="histograma_utilidad_con_tcero.png")

sys.exit()


def graficar_histogramas(data_1, data_2, x, bins, label_1, label_2, title, xlabel, ylabel="Conteo", ruta=""):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data_1, x=x, bins=bins, label=label_1)
    sns.histplot(data=data_2, x=x, bins=bins, label=label_2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(ruta)
    plt.close()
    # plt.show()
    # pass


graficar_histogramas(datos_escenarios_s_s,
                     datos_escenarios_r_q,
                     "utilidad_total",
                     30,
                     "Histograma utilidad (T, s, S)",
                     "Histograma utilidad (T, r, Q)",
                     "Utilidad distintas políticas en distintos escenarios",
                     "Utilidad total (en decenas de millones de pesos)",
                     ruta="histograma_utilidad.png")

graficar_histogramas(datos_escenarios_s_s,
                     datos_escenarios_r_q,
                     "quiebres_stock_total",
                     30,
                     "Quiebres stock (T, s, S)",
                     "Quiebres stock (T, r, Q)",
                     "Quiebres stock políticas en distintos escenarios",
                     "Quiebres stock",
                     ruta="histograma_quiebres_stock.png")

graficar_histogramas(datos_escenarios_s_s,
                     datos_escenarios_r_q,
                     "demanda_perdida_total",
                     30,
                     "Histograma demanda perdida (T, s, S)",
                     "Histograma demanda perdida (T, r, Q)",
                     "Demanda perdida políticas en distintos escenarios",
                     "Demanda perdida",
                     ruta="histograma_demanda_perdida.png")

graficar_histogramas(datos_escenarios_s_s,
                     datos_escenarios_r_q,
                     "costo_alm_total",
                     30,
                     "Costo almacenamiento (T, s, S)",
                     "Costo almacenamiento (T, r, Q)",
                     "Costo almacenamiento políticas en distintos escenarios",
                     "Costo almacenamiento (cientos de millones de pesos)",
                     ruta="histograma_costo_almacenamiento.png")


# graficar_histogramas(datos_escenarios_s_s,
#  datos_escenarios_r_q,
#  "ventas_totales",
#  30,
#  "Histograma ventas totales (T, s, S)",
#  "Histograma ventas totales (T, r, Q)",
#  "Ventas totales políticas en distintos escenarios",
#  "Ventas totales",
#  ruta="histograma_ventas_totales.png")


# sns.histplot(data=datos_escenarios_s_s, x="utilidad_total", bins=30, label="Histograma utilidad (T, s, S)")
# sns.histplot(data=datos_escenarios_r_q, x="utilidad_total", bins=30, label="Histograma utilidad (T, r, Q)")
# plt.title("Utilidad distintas políticas en distintos escenarios")
# plt.xlabel(r"Utilidad total (en decenas de millones de pesos)")
# plt.ylabel("Conteo")
# plt.legend()
# plt.show()
