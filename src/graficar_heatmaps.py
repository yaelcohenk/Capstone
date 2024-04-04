import pandas as pd
import os
import calplot
import matplotlib.pyplot as plt
from funciones.plotting import graficar_heatmap_datos


if False:
    graficar_heatmap_datos(os.path.join("datos", "ventas_diarias_subgrupos.xlsx"),
                           "Fecha",
                           "Subcategoría",
                           "subgrupo",
                           "Cantidad",
                           os.path.join("plots", "heatmaps", "subgrupos_diario"))

if False:
    graficar_heatmap_datos(os.path.join("datos", "ventas_diarias_grupos.xlsx"),
                           "Fecha",
                           "Grupo",
                           "grupo",
                           "Ventas",
                           os.path.join("plots", "heatmaps", "grupos_diario"))


# Reemplazaar nombres con / , """ .
if False:
    graficar_heatmap_datos(os.path.join("datos", "ventas_diarias_productos.xlsx"),
                           "Fecha",
                           "Descripción",
                           "grupo",
                           "Cantidad",
                           os.path.join("plots", "heatmaps", "productos_diario"))
