y = {0: 20}



x = {1: 0,
     2: 0,
     3: 5,
     4: 0,
     5: 0}


z = {1: 0,
     2: 0,
     3: 1,
     4: 0,
     5: 0}

w = {1: 3,
     2: 2,
     3: 3,
     4: 2,
     5: 2}


l = 2

for t in range(1, 6):
    valor_y = f"y_{t}"
    nuevo_valor_y = y.get(t - 1, 0) - w[t] + x.get(t - l, 0) * z.get(t - l, 0)
    if t not in y.keys():
        y[t] = nuevo_valor_y
    print(f"El valor del inventario al final de t = {t} es {nuevo_valor_y}") 
