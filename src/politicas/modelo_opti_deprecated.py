import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Crear un nuevo modelo
modelo = gp.Model("MiModelo")
items_df= pd.read_csv("data_items.csv", sep=";")
J=set(items_df["item_id"]) #Conjunto productos

T={1,2,3,4} #Conjunto tiempo

#Parametros
demanda = {(1, 1): 50, (1, 2): 60, (1, 3): 70, (1, 4): 80,  # Demanda del producto 1 en cada periodo
           (2, 1): 30, (2, 2): 35, (2, 3): 40, (2, 4): 45,  # Demanda del producto 2 en cada periodo
           (3, 1): 40, (3, 2): 45, (3, 3): 50, (3, 4): 55} 
precio_venta_producto = items_df.set_index('item_id')['unit_sale_price (CLP)'].to_dict()
costo_producto = items_df.set_index('item_id')['cost (CLP)'].to_dict()
costo_fijo_compra_prod= items_df.set_index('item_id')['cost_per_purchase'].to_dict()
precio_almacenamiento_por_dia_producto= items_df.set_index('item_id')['storage_cost (CLP)'].to_dict()
leadtime_producto= items_df.set_index('item_id')['leadtime (days)'].to_dict()
volumen_utilizado_producto= items_df.set_index('item_id')['size_m3'].to_dict()
volumen_maximo_bodega = 125


# Definir variables de decisión
x= modelo.addVars(J,T, vtype=GRB.CONTINUOUS, name=f"CANTIDAD COMPRADA PROD J EN TIEMPO T ") #CANTIDAD COMPRADA PROD J EN TIEMPO T 
y= modelo.addVars(J,T, vtype=GRB.CONTINUOUS, name=f"NVENTARIO PROD J EN TIEMPO T") # INVENTARIO PROD J EN TIEMPO T
z= modelo.addVars(J,T, vtype=GRB.BINARY, name=f"SI SE COMPRA PROD J EN TIEMPO T") #SI SE COMPRA PROD J EN TIEMPO T
w= modelo.addVars(J,T, vtype=GRB.CONTINUOUS, name=f"CANTIDAD VENDER PROD J EN INTERVALO T") #CANTIDAD VENDER PROD J EN INTERVALO T
h= modelo.addVars(J,T, vtype=GRB.CONTINUOUS, name=f"VAR AUX") #VAR AUX

# Establecer la función objetivo
modelo.setObjective(quicksum(quicksum((w[j,t]*precio_venta_producto[j]) for j in J )for t in T)- quicksum(quicksum((x[j,t]*costo_producto[j]+z[j,t]*costo_fijo_compra_prod[j]+y[j,t]*precio_almacenamiento_por_dia_producto[j]) for j in J )for t in T), GRB.MAXIMIZE)

# Agregar restricciones
modelo.addConstr(((quicksum((volumen_utilizado_producto[j]*y[j,t]) for j in J) <= volumen_maximo_bodega) for t in T), "No superar cap maxima")
modelo.addConstr((((x[j,t]<=1000000*z[j,t]) for j in J ) for t in T ), "se compra producto solo si se decide comprar")
modelo.addConstr((((w[j,t]>=demanda[j,t]) for j in J ) for t in T ), "satisface demanda")
modelo.addConstr((((y[j,t+1]==y[j,t]+ quicksum((w[j,t]+h[j,t])for t in T)) for j in J )), "inventario")
modelo.addConstr((((h[j,t]<=x[j,t]) for j in J ) for t in T ), "inventario")
modelo.addConstr((((h[j,t]<=10000*z[j,t]) for j in J ) for t in T ), "inventario")
modelo.addConstr((((h[j,t]>=x[j,t]+10000*(z[j,t]-1)) for j in J ) for t in T ), "inventario")
modelo.addConstr((((h[j,t]<=x[j,t]) for j in J ) for t in T ), "inventario")
modelo.addConstr((((h[j,t]>=0) for j in J ) for t in T ), "Naturaleza")
modelo.addConstr((((x[j,t]>=0) for j in J ) for t in T ), "Naturaleza")
modelo.addConstr((((w[j,t]>=0) for j in J ) for t in T ), "Naturaleza")
modelo.addConstr((((y[j,t]>=0) for j in J ) for t in T ), "Naturaleza")




# Optimizar el modelo
modelo.optimize()

# Imprimir la solución óptima
if modelo.status == GRB.OPTIMAL:
    print(f"Solución óptima: x = {x.x}, y = {y.x}")
else:
    print("No se encontró una solución óptima.")