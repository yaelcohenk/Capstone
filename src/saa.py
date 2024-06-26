import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
import random
import numpy as np

from datetime import timedelta
from gurobipy import Model, GRB, quicksum
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES

datos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
datos = datos[["Descripción", "Fecha", "Cantidad"]]

datos = datos[datos["Fecha"].dt.year >= 2023]

fecha_min = pd.Timestamp(year=2023, month=1, day=1)
fecha_max = max(datos["Fecha"])

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

with open("diccionario_params_productos.pkl", "rb") as f:
    diccionario_prods_params = pickle.load(f)


for producto, valores in diccionario_prods_params.items():
    precio_venta, costo_compra, costo_almacenar, leadtime, volumen, costo_fijo_comprar = valores
    v[producto] = precio_venta
    c[producto] = costo_compra
    CF[producto] = costo_fijo_comprar
    alpha[producto] = costo_almacenar
    l[producto] = leadtime
    Vol[producto] = volumen


K = range(30)  # Número escenarios

D_k = {}


with open("distrs.pkl", "rb") as file:
    distrs = pickle.load(file)


for key, value in D.items():
    distr_producto = distrs[key[0]]
    mu, std = distr_producto

    for k in K:
        nombre, t = key
        D_k[nombre, t, k] = max(0, value + np.random.normal(mu, std))


Vmax = 120
model = Model()
model.setParam("TimeLimit", 60 * 10)


x = model.addVars(J, T, name="x")
y = model.addVars(J, T, K, name="y", lb=-GRB.INFINITY)
y_plus = model.addVars(J, T, K, name="y+")
y_minus = model.addVars(J, T, K, name="y-")
z = model.addVars(J, T, vtype=GRB.BINARY, name="z")
w = model.addVars(J, T, K, name="w")

model.update()

M = 10 ** 5

model.addConstrs(x[j, t] <= M * z[j, t] for j in J for t in T)

model.addConstrs(quicksum(Vol[j] * y_plus[j, t, k]
                 for j in J) <= Vmax for t in T for k in K)

model.addConstrs(w[j, t, k] <= D_k.get((j, t, k), 0)
                 for j in J for t in T for k in K)

# Hasta acá funciona joya
model.addConstrs(y[j, t, k] == y_plus[j, t, k] - y_minus[j, t, k]
                 for j in J for t in T for k in K)

model.addConstrs(y[j, t, k] == y[j, t - timedelta(days=1), k] - w[j, t, k] +
                 x.get((j, t - timedelta(days=l[j])), 0) for j in J for t in T[1:] for k in K)

# Esto hay que cambiarlo creo
model.addConstrs(y[j, T[0], k] == 0 for j in J for k in K)


# Hay que poner la restricción de solo comprar cada 7 días
model.addConstrs(z[j, t] == 0 for j in J for indice_t,
                 t in enumerate(T) if indice_t % 7 != 0)


# Ver si la función objetivo es realmente la misma, da distinto esto a la política t, s, S
# model.setObjective(quicksum(v[j] * w[j, t, k] - c[j] * x[j, t] - CF[j] * z[j, t] - alpha[j]
#    * y_plus[j, t] - (v[j] - c[j]) * y_minus[j, t] for j in J for t in T), GRB.MAXIMIZE)

model.setObjective((1 / len(K)) * quicksum(v[j] * w[j, t, k] - alpha[j] * y_plus[j, t, k] - (
    v[j] - c[j]) * y_minus[j, t, k] for j in J for t in T for k in K) - quicksum(c[j] * x[j, t] + CF[j] * z[j, t] for j in J for t in T), GRB.MAXIMIZE)


model.optimize()



print(model.ObjVal)


productos_vendidos_promedio = np.mean(x.X for x in w.values())
ordenes_compra_promedio = np.mean(x.X for x in z.values())
cantidad_quiebres_stock_promedio = []
unidades_quebradas_promedio = []


for valor in y_minus.values():
    if valor.X > 0:
        cantidad_quiebres_stock_promedio.append(1)
        unidades_quebradas_promedio.append(valor.X)


cantidad_quiebres_stock_promedio = np.mean(cantidad_quiebres_stock_promedio)
unidades_quebradas_promedio = np.mean(unidades_quebradas_promedio)
productos_comprados_promedio = np.mean(x.X for x in x.values())
utilidades = model.ObjVal

print(f"Se vendieron un total de {productos_vendidos_promedio} productos en promedio de productos")
print(f"En total se realizaron {ordenes_compra_promedio} ordenes de compra en promedio")
print(f"Hubo un total de {cantidad_quiebres_stock_promedio} quiebres de stock en promedio")
print(f"Se compraron un total de {productos_comprados_promedio} productos en promedio")
print(f"Las utilidades corresponden a {utilidades} CLP")




# cantidad_quiebres_stock_promedio = 
