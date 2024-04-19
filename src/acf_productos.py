import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sys

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Hacer el ACF para cada uno de los productos y luego agruparlos con ese modelo de 
# imágenes, para ver que productos tienen un ACF parecido. Así voy a poder discriminar modelos

productos_vigentes = pd.read_excel(os.path.join("datos", "prod_vigentes.xlsx"))
productos_vigentes = productos_vigentes["Descripción"].tolist()

ventas_productos = pd.read_excel(os.path.join("datos", "ventas_diarias_productos.xlsx"))
ventas_totales_prods = ventas_productos.groupby("Descripción").agg({"Cantidad": "sum"})

ventas_totales_prods = ventas_totales_prods[ventas_totales_prods["Cantidad"] > 31]
mas_30_vendidos = ventas_totales_prods.index.to_list()

contador = 0

# Hacer refactor de este código. Pasar todo a una función y listo
for producto in productos_vigentes:
    if producto in mas_30_vendidos:
        try:
            producto_actual = ventas_productos[ventas_productos["Descripción"].isin([producto])]
            plotted = plot_acf(producto_actual["Cantidad"], lags=30)
            plt.xlabel("Lags")
            plt.ylabel("Autocorrelación")
            plt.title(f"ACF para producto {producto}")
            plotted.savefig(os.path.join("plots", "acf_productos_vigentes", f"acf_{producto}.png"))
            plt.close()
        except FileNotFoundError:
            producto_actual = ventas_productos[ventas_productos["Descripción"].isin([producto])]
            plotted = plot_acf(producto_actual["Cantidad"], lags=30)
            plt.xlabel("Lags")
            plt.ylabel("Autocorrelación")
            plt.title(f"ACF para producto {producto}")
            plotted.savefig(os.path.join("plots", "acf_productos_vigentes", f"acf_prod_{contador}.png"))
            plt.close()
            contador += 1
        except ValueError:
            producto_actual = ventas_productos[ventas_productos["Descripción"].isin([producto])]
            plotted = plot_acf(producto_actual["Cantidad"]) # Aquí no especifique los lags
            plt.xlabel("Lags")
            plt.ylabel("Autocorrelación")
            plt.title(f"ACF para producto {producto}")
            plotted.savefig(os.path.join("plots", "acf_productos_vigentes", f"acf_prod_{contador}.png"))
            plt.close()
            contador += 1

