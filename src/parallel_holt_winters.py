import pandas as pd
import ray
import optuna

from parametros import (PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@ray.remote
def holt_winter(venta_productos):
    study = optuna.create_study(direction="minimize")

    pass

    def optimizar_holt_winters(trial, cantidad=venta_productos):
        smoothing_level = trial.suggest_uniform('alpha', 0, 1)
        smoothing_trend = trial.suggest_uniform('beta', 0, 1)
        smoothing_seasonal = trial.suggest_uniform('gamma', 1 - smoothing_level, 1)

        prediccion = ExponentialSmoothing(cantidad,
                                          trend="add",
                                          seasonal="add",
                                          seasonal_periods=7)


if __name__ == '__main__':
    ventas_productos = pd.read_excel(
        PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos.index = ventas_productos["Fecha"]
    ventas_productos.drop("Fecha", axis=1, inplace=True)

    productos = ventas_productos["Descripción"].unique().tolist()

    cantidades_productos = list()

    for producto in productos:
        producto_loop = ventas_productos[ventas_productos["Descripción"].isin([
                                                                              producto])]
        producto_loop = producto_loop["Cantidad"].to_numpy()
        cantidades_productos.append(producto_loop)
