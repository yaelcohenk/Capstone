import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


from datetime import timedelta
from gurobipy import Model, GRB, quicksum
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS

datos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS)
datos = datos[["Descripción", "Fecha", "Cantidad"]]

datos_ganancias = pd.read_excel(os.path.join("datos", "ganancias_total_productos.xlsx"))
datos_ganancias = datos_ganancias.head(75)
productos = datos_ganancias["description"]

datos = datos[datos["Descripción"].isin(productos)]
datos = datos[datos["Fecha"].dt.year >= 2023]


# print(datos)

# sys.exit("EXIT")

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

model.setParam("TimeLimit", 30)


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


# Hay que poner la restricción de solo comprar cada 7 días
model.addConstrs(z[j, t] == 0 for j in J for indice_t, t in enumerate(T) if indice_t % 7 != 0)


model.setObjective(quicksum(v[j] * w[j, t] - c[j] * x[j, t] - CF[j] * z[j, t] - alpha[j]
                   * y_plus[j, t] - (v[j] - c[j]) * y_minus[j, t] for j in J for t in T), GRB.MAXIMIZE)



model.optimize()


# Quizás todo esto de abajo lo tengo que pasar a algunas otras funciones cuando pueda para que quede más bonito

cantidad_quiebres_stock = 0
unidades_quebradas = 0


for valor in y_minus.values():
    if valor.X != 0:
        cantidad_quiebres_stock += 1
        unidades_quebradas += valor.X



productos_vendidos = 0
for valor in w.values():
    productos_vendidos += valor.X


ordenes_de_compra = 0

for valor in z.values():
    ordenes_de_compra += valor.X

productos_comprados = 0

for valor in x.values():
    productos_comprados += valor.X

print(f"Se vendieron un total de {productos_vendidos} de producto")
print(f"En total se realizaron {ordenes_de_compra} ordenes de compra")
print(f"Considerando todos los productos, hubo un total de {cantidad_quiebres_stock} quiebres de stock")
print(f"Hubo un quiebre de stock por un total de {unidades_quebradas} de stock")
print(f"Se compraron un total de {productos_comprados} productos")

print(f"Las utilidades corresponden a {model.ObjVal} CLP")



tiempo = []
inventario = []

inventario_producto = dict()


for t in T:
    inventario_t = 0
    
    for j in J:
        inventario_t += y[j, t].X

    tiempo.append(t)
    inventario.append(inventario_t)


for j in J:
    inventario_producto_tiempo = list()

    for t in T:
        inventario_producto_tiempo.append(y[j, t].X)

    inventario_producto[j] = inventario_producto_tiempo

    


sns.lineplot(x=tiempo,y=inventario)
plt.xlabel("Fechas")
plt.ylabel("Cantidad de inventario (unidades)")
plt.title(f"Inventario a través del tiempo del sistema")
plt.savefig(os.path.join("politicas_graficos", "inventario", "modelo_opti", f"inventario_sistema.png"))
plt.close()



contador = 0
mapeo_graficos = dict()

for producto, inventario_prod in inventario_producto.items():
    sns.lineplot(x=tiempo, y=inventario_prod)
    plt.xlabel("Fechas")
    plt.ylabel("Cantidad de inventario (unidades)")
    plt.title(f"Inventario a través del tiempo para {producto}")
    plt.savefig(os.path.join("politicas_graficos", "inventario", "modelo_opti", f"prod_{contador}.png"))
    plt.close()

    mapeo_graficos[contador] = producto
    contador += 1


with open(os.path.join("politicas_graficos", "inventario", "mapeos.txt"), "w") as file:
    json.dump(mapeo_graficos, file)

# 1) Ventas totales unidades
# 2) Total de órdenes realizadas
# 3) Total de quiebres de stock
# 4) Cantidad de demanda perdida total
# 5) Cantidad de productos comprados
# 6) Utilidades