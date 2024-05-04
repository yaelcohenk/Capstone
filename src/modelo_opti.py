import pandas as pd
import os
import sys

from datetime import timedelta
from gurobipy import Model, GRB, quicksum
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS

datos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS)
datos = datos[["Descripción", "Fecha", "Cantidad"]]

datos_ganancias = pd.read_excel(os.path.join("datos", "ganancias_total_productos.xlsx"))
datos_ganancias = datos_ganancias.head(75)
productos = datos_ganancias["description"]

datos = datos[datos["Descripción"].isin(productos)]

fecha_min = min(datos["Fecha"])
fecha_max = max(datos["Fecha"])

# Necesitamos tener todos los días entre medio, no solo los que registraron venta
T = [fecha_min + timedelta(days=i)
     for i in range((fecha_max - fecha_min).days + 1)]


J = datos["Descripción"].unique().tolist()
D = dict()

datos_lista = datos.to_numpy()
for nombre, t, demanda in datos_lista:
    D[nombre, t] = demanda


v = dict()          # Precio de venta del producto
c = dict()          # Costo de comprar el producto
CF = dict()         # Costo fijo de comprar
alpha = dict()      # Costo almacenamiento
l = dict()          # Leadtime en días
Vol = dict()        # Volumen utilizado


datos_productos = pd.read_excel(os.path.join("datos", "data_items_fillna.xlsx"))
columnas = ["description",
            "unit_sale_price (CLP)",
            "cost (CLP)",
            "storage_cost (CLP)",
            "leadtime (days)",
            "size_m3",
            "cost_per_purchase"]

datos_productos = datos_productos[datos_productos["description"].isin(J)]
datos_productos = datos_productos[columnas].drop_duplicates().to_numpy()


for prod in datos_productos:
    descripcion, precio_venta, costo, almacenamiento_precio, leadtime, vol, costo_fijo = prod
    v[descripcion] = precio_venta
    c[descripcion] = costo
    CF[descripcion] = costo_fijo
    alpha[descripcion] = almacenamiento_precio
    l[descripcion] = leadtime
    Vol[descripcion] = vol


Vmax = 120

model = Model()
x = model.addVars(J, T, name="x")
y = model.addVars(J, T, name="y", lb=-GRB.INFINITY)
y_plus = model.addVars(J, T, name="y+")
y_minus = model.addVars(J, T, name="y-")
z = model.addVars(J, T, vtype=GRB.BINARY, name="z")
w = model.addVars(J, T, name="w")

model.update()

M = 10 ** 5

model.addConstrs(x[j, t] <= M * z[j, t] for j in J for t in T)

model.addConstrs(quicksum(Vol[j] * y_plus[j, t] for j in J) <= Vmax for t in T)

model.addConstrs(w[j, t] <= D.get((j, t), 0) for j in J for t in T)

# Hasta acá funciona joya
model.addConstrs(y[j, t] == y_plus[j, t] - y_minus[j, t] for j in J for t in T)

model.addConstrs(y[j, t] == y[j, t - timedelta(days=1)] - w[j, t] +
                 x.get((j, t - timedelta(days=l[j])), 0) for j in J for t in T[1:])

model.addConstrs(y[j, T[0]] == 0 for j in J)  # Esto hay que cambiarlo creo

model.setObjective(quicksum(v[j] * w[j, t] - c[j] * x[j, t] - CF[j] * z[j, t] - alpha[j]
                   * y_plus[j, t] - (v[j] - c[j]) * y_minus[j, t] for j in J for t in T), GRB.MAXIMIZE)


# Hay que poner la restricción de solo comprar cada 7 días

model.optimize()
