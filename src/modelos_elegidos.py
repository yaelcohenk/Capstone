import pandas as pd
import os
import sys



datos = pd.read_excel(os.path.join("datos", "Selección_modelo.xlsx"))
# datos_1 = pd.read_excel(os.path.join("datos", "Selección_modelo_1.xlsx"))
print(datos.groupby("Modelo de pronostico").count().reset_index())


sys.exit()
# print(datos)
# print(datos_1)
descripciones = datos["Descripción"].tolist()
descripciones_1 = datos_1["Descripción"].tolist()

# print(descripciones)

no_estan = list()
for descripcion in descripciones_1:
    if descripcion not in descripciones:
        print(descripcion)
        no_estan.append(descripcion)


# print(no_estan)
print(datos_1[datos_1["Descripción"].isin(no_estan)])