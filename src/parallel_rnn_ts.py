import os
import sys
import pandas as pd
import ray


from parametros import (PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)




def rnn(dataframe_producto: pd.DataFrame, nombre_producto: str):
    dataframe = dataframe_producto.drop("Descripción", axis=1)
    dataframe = dataframe.drop("Fecha", axis=1) 

    dataframe.dropna(inplace=True)

    columnas_covariables = [columna for columna in dataframe.columns if columna != "Cantidad"]
    


    
    pass



if __name__ == '__main__':
    ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos.set_index("Fecha", inplace=True)
    
    productos = ventas_productos["Descripción"].unique().tolist()[:1]
    
    lista_productos = [ventas_productos[ventas_productos["Descripción"].isin([i])] for i in productos]
    lista_final = list(zip(lista_productos, productos))


