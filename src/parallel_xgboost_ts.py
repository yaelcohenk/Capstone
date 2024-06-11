import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import ray
import sys
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error,
                             mean_absolute_error,
                             mean_squared_error)
from sklearn.preprocessing import StandardScaler

from parametros import (FEATURES,
                        TARGET,
                        PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)


@ray.remote
def xgboost_producto(dataframe_producto: pd.DataFrame,
                     nombre_producto: str = "",
                     random_state=42):

    try:
        entrenamiento = dataframe_producto[dataframe_producto["year"] < 2023]
        validacion = dataframe_producto[dataframe_producto["year"] >= 2023]

        X_train, y_train = entrenamiento[FEATURES], entrenamiento[TARGET]
        scaler_trainer = StandardScaler()
        scaler_values = StandardScaler()

        x_train_scaled = scaler_trainer.fit_transform(np.array(X_train))
        y_train_scaled = scaler_values.fit_transform(
            np.array(y_train).reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(x_train_scaled, y_train_scaled,
                                                            test_size=0.2,
                                                            random_state=random_state)

        reg = xgb.XGBRegressor(n_estimators=1000, objective="reg:squarederror")

        reg.fit(X_train, y_train, eval_set=[
                (X_train, y_train), (X_test, y_test)], verbose=100)

        X_val_1, y_val = validacion[FEATURES], validacion[TARGET]
        X_val = scaler_trainer.transform(np.array(X_val_1))
        y_val = scaler_values.transform(np.array(y_val).reshape(-1, 1))

        # La validación la deberíamos hacer sobre los datos 2023-2024
        prediction = reg.predict(X_val).squeeze()
        prediction = scaler_values.inverse_transform(prediction.reshape(-1, 1)).flatten()
        y_val = scaler_values.inverse_transform(y_val.reshape(-1, 1)).flatten()
        errores=[]
        suma=0
        for i in range( len(prediction)):
            error=y_val[i]-prediction[i]
            suma+=error
            errores.append(error)
        MAD_prophet=np.mean(np.abs(errores))
        if MAD_prophet!=0:
            tracking_signal=suma/MAD_prophet
        else:
            tracking_signal = np.nan

        # Cualquier modelo con información estable un modelo lo va a captar bien
        # Ver como presentamos esos productos con demanda constante

        # Para productos con error 0, podríamos generar una política mucho mejor que todo
        # el resto. No podemos confirmarlo, pero si inferirlo, sería valioso tener esa información
        #

        # Acá abajo puedo poner más métricas
        mape = mean_absolute_percentage_error(y_val, prediction)
        rmse = root_mean_squared_error(y_val, prediction)
        mae = mean_absolute_error(y_val, prediction)
        mse = mean_squared_error(y_val, prediction)

        return nombre_producto, mape, rmse, mae, mse,tracking_signal, y_val, prediction, X_val_1

        # Revisar el valor del MAPE si está en % o no. Mirar de manera crítica los indicadores

        # Ver que es un cliente fiel, hacer propuesta que puede ser arbitraria pero no random
        # Puede ser razonable en esta industria que compre todos los meses. Depende como se 
        # dividen los clientes entre si. Podríamos hacer un clustering, análisis de patrones de compra


        # Podríamos tomar hasta 10 productos, tiene que quedar claro que vamos a hacer con el resto
        # Obtener resultados preliminares para un grupo de los productos

        # Se espera tener un prototipo funcional para versión simplificada del problema. No tomar el universo
        # completo, sino tomar un grupo de todo eso. Es válido hacerlo para un grupo, no para todos

        # Si dejamos productos de lado, es porque el beneficio que se obtiene es más bajo que el costo.
        # Si tenemos productos que aportan poquita plata y son parecidos a otros productos que implementé política
        # y tenemos el resto automatizado, podemos intentar implementarlo, ya que sería rápido

        # Habría que calcular cuanto % de ganancias está en esos 300 productos.

        # Puede ser un argumento dejar los datos fuera, ya que el pronóstico puede ser tan malo
        # que no nos sirve.

    except ValueError as e:
        print(f"Ha ocurrido un error {e}")
        return "ValueError"


if __name__ == '__main__':
    ray.init()
    ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos.index = ventas_productos["Fecha"]
    ventas_productos.drop("Fecha", axis=1, inplace=True)
    productos = ventas_productos["Descripción"].unique().tolist()

    lista_productos = [ventas_productos[ventas_productos["Descripción"].isin([i])] for i in productos]

    lista_final = list(zip(lista_productos, productos))

    futures = [xgboost_producto.remote(dataframe_prod, prod) for dataframe_prod, prod in lista_final]

    elementos = ray.get(futures)

    valores = list()
    predicciones = dict()

    for elemento in elementos:
        nombre, *metricas, y_val, y_pred, covariables = elemento
        
        valores.append([nombre, *metricas])
        predicciones[nombre] = [y_val.tolist(), y_pred.tolist(), covariables.index]

    dataframe = pd.DataFrame(valores)
    dataframe.columns = ["producto", "MAPE", "RMSE", "MAE", "MSE", "Tracking Signal"]
    dataframe.set_index("producto", inplace=True)
    dataframe.to_excel(os.path.join("datos", "metricas_xgboost.xlsx"))




    contador = 0
    diccionario_equivalencias_nombres = dict()
    for nombre, values in predicciones.items():
        dataframe = pd.DataFrame({"valor_real": values[0], "valor_prediccion": values[1]})
        dataframe.set_index(values[2], inplace=True) 
        dataframe.to_excel(os.path.join("predicciones", "xgboost", f"producto_{contador}.xlsx"))
        diccionario_equivalencias_nombres[contador] = nombre
        contador += 1

     
    with open(os.path.join("predicciones", "xgboost", "mapeos.txt"), "w") as file:
        json.dump(diccionario_equivalencias_nombres, file)


    sys.exit("[INFO]: EJECUCIÓN EXITOSA PARALLEL XGBOOST")
    for elemento in predicciones:
        dato_loop = pd.DataFrame({"prediction": elemento[1].tolist(), "real": elemento[2].tolist()})
        print(dato_loop)

        dato_loop['real'].plot(style='b', figsize=(10, 5), label='Original')
        dato_loop['prediction'].plot(style='r', figsize=(10, 5), label='Predicción')

        plt.xlabel('Fecha')
        plt.ylabel('Cantidad')
        plt.title('Predicciones XGBoost: Comparación de Serie Original y Predicción')
        plt.legend()
        plt.show()

        plt.close()