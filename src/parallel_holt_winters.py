import pandas as pd
import ray
import optuna
import warnings
import numpy as np

from parametros import (PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from functools import partial
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error)

warnings.filterwarnings('ignore')

# @ray.remote
def holt_winters(dataframe_producto: pd.DataFrame, nombre: str = ""):

    study = optuna.create_study(direction="minimize")
    entrenamiento = dataframe_producto[dataframe_producto["year"] < 2023]
    validacion = dataframe_producto[dataframe_producto["year"] >= 2023]


    def optimizar_holt_winters(trial, entrenamiento, validacion):
        smoothing_level = trial.suggest_uniform('alpha', 0, 1)
        smoothing_trend = trial.suggest_uniform('beta', 0, 1)
        smoothing_seasonal = trial.suggest_uniform('seasonal', 1 - smoothing_level, 1)

        ventas_estimadas = (ExponentialSmoothing(entrenamiento["Cantidad"].to_numpy(),
                                                trend="add",
                                                seasonal="add",
                                                seasonal_periods=7).
                                                fit(smoothing_level=smoothing_level,
                                                    smoothing_trend=smoothing_trend,
                                                    smoothing_seasonal=smoothing_seasonal))
        


        # valores_obtenidos = ventas_estimadas.fittedvalues
        # rmse = root_mean_squared_error(valores_obtenidos, entrenamiento["Cantidad"].to_numpy())
        predicciones = ventas_estimadas.forecast(steps=len(validacion))
        mae = root_mean_squared_error(validacion["Cantidad"].to_numpy(), predicciones)

        # print(predicciones)
        # print(f"[INFO]: El MAE es {mae}")
        
        if np.isnan(mae):
            return float("inf")
        
        return mae
    
    objective = partial(optimizar_holt_winters, entrenamiento=entrenamiento, validacion=validacion)
    valores = study.optimize(objective, n_trials=100)

    best_params = study.best_params

    # print(f"Los mejores parámetros son {best_params}")
    predicciones = (ExponentialSmoothing(entrenamiento["Cantidad"].to_numpy(),
                                         trend="add",
                                         seasonal="add",
                                         seasonal_periods=7).
                                         fit(smoothing_level=best_params["alpha"],
                                             smoothing_trend=best_params["beta"],
                                             smoothing_seasonal=best_params["seasonal"]))
    # print
    predicciones = predicciones.forecast(steps=len(validacion))
    return best_params, predicciones, validacion["Cantidad"]
    # 
    # predicciones_sig = predicciones.forecast(steps=len(validacion))

    # return validacion["Cantidad"], predicciones_sig
    # return 1, 2

    


if __name__ == '__main__':
    ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos.set_index("Fecha", inplace=True)
    ventas_productos = ventas_productos[ventas_productos["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]
    ventas_productos.drop("Descripción", axis=1, inplace=True)


    best_params, predicciones, valores_reales = holt_winters(ventas_productos)
    # print(ventas_productos)
    # print(valores)
    print(best_params)
    print(predicciones, valores_reales)






# @ray.remote
# def holt_winter(venta_productos):
    # study = optuna.create_study(direction="minimize")
# 
    # pass
# 
    # def optimizar_holt_winters(trial, cantidad=venta_productos):
        # smoothing_level = trial.suggest_uniform('alpha', 0, 1)
        # smoothing_trend = trial.suggest_uniform('beta', 0, 1)
        # smoothing_seasonal = trial.suggest_uniform('gamma', 1 - smoothing_level, 1)
# 
        # prediccion = ExponentialSmoothing(cantidad,
                                        #   trend="add",
                                        #   seasonal="add",
                                        #   seasonal_periods=7)
# 
# 
# if __name__ == '__main__':
    # ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    # ventas_productos.index = ventas_productos["Fecha"]
    # ventas_productos.drop("Fecha", axis=1, inplace=True)
# 
    # productos = ventas_productos["Descripción"].unique().tolist()
# 
    # cantidades_productos = list()
# 
    # for producto in productos:
        # producto_loop = ventas_productos[ventas_productos["Descripción"].isin([
                                                                            #   producto])]
        # producto_loop = producto_loop["Cantidad"].to_numpy()
        # cantidades_productos.append(producto_loop)
