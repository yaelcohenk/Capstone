from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import sys
import numpy as np

import warnings
# optuna.logging.set_verbosity(optuna.logging.WARNING)
# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore')

ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
ventas_productos.index = ventas_productos["Fecha"]
ventas_productos = ventas_productos[ventas_productos["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]
ventas_productos.drop("Descripción", axis=1, inplace=True)
ventas_productos.drop("Fecha", axis=1, inplace=True)


ventas_productos = ventas_productos["Cantidad"].to_frame()

smoothing_level = 0.0002833446927321719
smoothing_trend = 0.5963922294249621
smoothing_seasonal = 0.4376467877455217

# ventas_productos["SESM"] = SimpleExpSmoothing(ventas_productos["Cantidad"]).fit().fittedvalues
ventas_productos["ES"] = (ExponentialSmoothing(ventas_productos["Cantidad"],
                                               trend="add",
                                               seasonal="add",
                                               seasonal_periods=7)
                          .fit(smoothing_level=smoothing_level,
                               smoothing_trend=smoothing_trend,
                               smoothing_seasonal=smoothing_seasonal
                               ).fittedvalues)


# sys.exit()
# print(type(ventas_productos))
print(ventas_productos)

plt.plot(ventas_productos["Cantidad"], color="blue", label="Valor real")
# plt.plot(ventas_productos["SESM"], color="red",
#  label="Predicción Simple Exponential Smoothing")
plt.plot(ventas_productos["ES"], color="red",
         label="Predicción Triple Exponential Smoothing")
plt.legend(loc="upper left")
plt.title("Holt Winter's Method with Trend and Seasonal")
plt.show()


sys.exit()

ventas_productos = ventas_productos["Cantidad"].to_numpy()
# print(ventas_productos, ventas_productos.shape)


def optimizar_parametros_holt_winters(trial, cantidades=ventas_productos):
    smoothing_level = trial.suggest_uniform('alpha', 0, 1)
    smoothing_trend = trial.suggest_uniform('beta', 0, 1)
    smoothing_seasonal = trial.suggest_uniform('gamma', 1 - smoothing_level, 1)

    ventas_estimadas = ExponentialSmoothing(cantidades,
                                            trend="add",
                                            seasonal="add",
                                            seasonal_periods=7).fit(smoothing_level=smoothing_level,
                                                                    smoothing_trend=smoothing_trend,
                                                                    smoothing_seasonal=smoothing_seasonal).fittedvalues

    valores_absolutos = np.abs(ventas_estimadas - cantidades)

    mad = np.mean(valores_absolutos)
    
    if np.isnan(mad):
        return float("inf")
    
    return mad


study = optuna.create_study(direction="minimize")

study.optimize(optimizar_parametros_holt_winters, n_trials=1000)

best_params = study.best_params
best_value = study.best_value

print(f"Los mejores parámetros son {best_params}")
print(f"El mejor valor obtenido es {best_value}")
