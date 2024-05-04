import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
import sys


datos_diarios_grupos = pd.read_excel(os.path.join("datos", "ventas_diarias_subgrupos.xlsx"))

datos_diarios_grupos.columns = ["Grupo", "Fecha", "Ventas"]

lista_grupos_grandes = datos_diarios_grupos["Grupo"].unique()

for grupo in lista_grupos_grandes:
    try:

        datos_diarios_accesorios = datos_diarios_grupos[datos_diarios_grupos["Grupo"] == grupo]
        datos_diarios_accesorios = datos_diarios_accesorios.set_index("Fecha")
        datos_diarios_accesorios = datos_diarios_accesorios["Ventas"]

        decomposition = sm.tsa.seasonal_decompose(
            datos_diarios_accesorios, period=7)
        fig = decomposition.plot()
        fig.set_size_inches((16, 9))
        fig.suptitle(
            f'Información ventas diarias del subgrupo {grupo}', fontsize=20)
        fig.tight_layout()
        fig.savefig(os.path.join("plots", "seasonal_plots",
                    "subgrupos_diario", f"informaciones_diarias_{grupo}.png"))

        plt.close()
    except Exception as e:
        print(e)
