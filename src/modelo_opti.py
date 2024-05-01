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

# Estaríamos observando los datos de las ventas de los 75 productos más vendidos nomás
# print(datos)

# T = datos["Fecha"].to_list()  # Fechas demanda
# D = datos["Cantidad"].to_list()  # Demandas producto en cada fecha
# J = datos["Descripción"].to_list()  # Productos en cuestión
# print(min(datos["Fecha"]), max(datos["Fecha"]))
fecha_min = min(datos["Fecha"])
fecha_max = max(datos["Fecha"])


# Necesitamos tener todos los días entre medio, no solo los que registraron venta
T = [fecha_min + timedelta(days=i) for i in range((fecha_max - fecha_min).days + 1)]
# print(lista_dias)


# sys.exit()
# T = datos["Fecha"].unique().tolist()
J = datos["Descripción"].unique().tolist()
D = dict()

datos_lista = datos.to_numpy()
for nombre, t, demanda in datos_lista:
    D[nombre, t] = demanda

# print(D)

# print(datos_lista)
# print(len(datos["Descripción"].unique().tolist())) #

# sys.exit()
# print(len(J))

v = dict() # Precio de venta del producto
c = dict() # Costo de comprar el producto
CF = dict() # Costo fijo de comprar
alpha = dict() # Costo almacenamiento
l = dict() # Leadtime en días
Vol = dict() # Volumen utilizado


datos_productos = pd.read_csv(os.path.join("datos", "data_items.csv"), sep=";")
columnas = ["description",
            "unit_sale_price (CLP)",
            "cost (CLP)",
            "storage_cost (CLP)",
            "leadtime (days)",
            "size_m3",
            "cost_per_purchase"]

datos_productos = datos_productos[datos_productos["description"].isin(J)]

media_precio = datos_productos["unit_sale_price (CLP)"].mean()
media_costo = datos_productos["cost (CLP)"].mean()
media_costo_alm = datos_productos["storage_cost (CLP)"].mean()
media_leadtime = datos_productos["leadtime (days)"].mean()
media_vol = datos_productos["size_m3"].mean()
media_cost_compra = datos_productos["cost_per_purchase"].mean()

# Esto lo tengo que pasar a otro archivo. Esto no hay que hacerlo aquí, se mezcla la lógica
datos_productos["unit_sale_price (CLP)"].fillna(media_precio, inplace=True)
datos_productos["cost (CLP)"].fillna(media_costo, inplace=True)
datos_productos["storage_cost (CLP)"].fillna(media_costo_alm, inplace=True)
datos_productos["leadtime (days)"].fillna(media_leadtime, inplace=True)
datos_productos["size_m3"].fillna(media_vol, inplace=True)
datos_productos["cost_per_purchase"].fillna(media_cost_compra, inplace=True)


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
z = model.addVars(J,T, vtype=GRB.BINARY, name="z")
w = model.addVars(J, T, name="w")

model.update()

# print(x.get((-1, 1), 0))

# sys.exit()
M = 10 ** 5

model.addConstrs(x[j, t] <= M * z[j, t] for j in J for t in T)

model.addConstrs(quicksum(Vol[j] * y_plus[j, t] for j in J) <= Vmax for t in T)

model.addConstrs(w[j, t] <= D.get((j, t), 0) for j in J for t in T)

# Hasta acá funciona joya
model.addConstrs(y[j, t] == y_plus[j, t] - y_minus[j, t] for j in J for t in T)

model.addConstrs(y[j, t] == y[j, t - timedelta(days=1)] - w[j, t] + x.get((j, t - timedelta(days=l[j])), 0) for j in J for t in T[1:])

model.addConstrs(y[j, T[0]] == 0 for j in J) # Esto hay que cambiarlo creo

model.setObjective(quicksum(v[j] * w[j, t] - c[j] * x[j, t] - CF[j] * z[j, t] - alpha[j] * y_plus[j, t] - (v[j] - c[j]) * y_minus[j, t] for j in J for t in T), GRB.MAXIMIZE)

model.optimize()

