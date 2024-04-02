import pandas as pd
import os
import calplot
import matplotlib.pyplot as plt


def graficar_heatmap_datos(ruta_datos,
                           nombre_columna_fecha,
                           agrupacion,
                           tipo_dato,
                           nombre_columna_cantidad,
                           carpeta_guardar):

    datos = pd.read_excel(ruta_datos)
    datos = datos.set_index(nombre_columna_fecha)

    descripciones = list(datos[agrupacion].unique())

    for clasificacion in descripciones:
        try:
            datos_loop = datos[datos[agrupacion].isin([clasificacion])]
            fig, ax = calplot.calplot(datos_loop[nombre_columna_cantidad],
                                      cmap="coolwarm",
                                      colorbar=True,
                                      yearlabel_kws={'fontname': 'sans-serif'},
                                      figsize=(10, 10))

            fig.suptitle(
                f"Heatmap de ventas diarias de {tipo_dato} {clasificacion}")
            fig.savefig(os.path.join(carpeta_guardar,
                        f"heatmap_diario_{clasificacion}.png"))
            plt.close()
        except Exception as e:
            print(e)


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


#Reemplazaar nombres con / , """ . 
graficar_heatmap_datos(os.path.join("datos", "ventas_diarias_productos.xlsx"),
                       "Fecha",
                       "Descripción",
                       "grupo",
                       "Cantidad",
                       os.path.join("plots", "heatmaps", "productos_diario"))

# datos = pd.read_excel(os.path.join("datos", "ventas_diarias_subgrupos.xlsx"))
# print(datos.info())
# datos = datos.set_index("Fecha")


# subcategorias = list(datos["Subcategoría"].unique())

# print(type(datos["Subcategoría"].unique()))

# for subcategoria in subcategorias:
# datos_loop = datos[datos["Subcategoría"].isin([subcategoria])]
# fig, ax = calplot.calplot(datos_loop["Cantidad"],
#   cmap="coolwarm",
#   colorbar=True,
#   yearlabel_kws={'fontname': 'sans-serif'},
#   figsize=(10, 10))
#
# fig.suptitle(f"Heatmap de ventas diarias del subgrupo {subcategoria}")
# fig.savefig(os.path.join("plots", "heatmaps", "subgrupos_diario",
# f"heatmap_diario_{subcategoria}.png"))
# plt.close()
# plt.show()
