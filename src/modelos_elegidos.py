import pandas as pd
import os


datos = pd.read_excel(os.path.join("datos", "Selección_modelo.xlsx"))
print(datos.groupby("Modelo de pronostico").count())