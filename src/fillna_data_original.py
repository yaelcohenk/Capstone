import pandas as pd
import os


data_items = pd.read_csv(os.path.join("datos", "data_items.csv"), sep=";")

# print(data_items)
# print(data_items.columns)

media_precio = data_items["unit_sale_price (CLP)"].mean()
media_costo = data_items["cost (CLP)"].mean()
media_costo_alm = data_items["storage_cost (CLP)"].mean()
media_leadtime = data_items["leadtime (days)"].mean()
media_vol = data_items["size_m3"].mean()
media_cost_compra = data_items["cost_per_purchase"].mean()

# Esto lo tengo que pasar a otro archivo. Esto no hay que hacerlo aquí, se mezcla la lógica
data_items["unit_sale_price (CLP)"].fillna(media_precio, inplace=True)
data_items["cost (CLP)"].fillna(media_costo, inplace=True)
data_items["storage_cost (CLP)"].fillna(media_costo_alm, inplace=True)
data_items["leadtime (days)"].fillna(media_leadtime, inplace=True)
data_items["size_m3"].fillna(media_vol, inplace=True)
data_items["cost_per_purchase"].fillna(media_cost_compra, inplace=True)


data_items.to_excel(os.path.join("datos", "data_items_fillna.xlsx"), index=False)
