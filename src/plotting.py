import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os


ganancias_total_productos = pd.read_excel(
    os.path.join("datos", "ganancias_total_productos.xlsx"))
# print(ganancias_total_productos.head(75))

ganancias_total_productos = ganancias_total_productos.head(75)

fig = px.treemap(data_frame=ganancias_total_productos,
                 path=["group_description", "description"], values="porcentaje_ingreso_marginal",
                 custom_data=["porcentaje_ingreso_marginal"])

fig.update_traces(textinfo="label+value")

# fig.show()
fig.write_image("imagen.svg")