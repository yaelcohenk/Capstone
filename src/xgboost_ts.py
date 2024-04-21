import pandas as pd
import seaborn as sns
import os
import sys
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from scipy.fft import fft

from funciones.features import (create_features,
                                apply_trend_and_seasonal,
                                apply_day_selling_differences,
                                apply_fourier_transform)

from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES


ventas_productos = pd.read_excel(
    PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
ventas_productos.index = ventas_productos["Fecha"]
ventas_productos.drop("Fecha", axis=1, inplace=True)
# print(ventas_productos)

productos_unicos = ventas_productos["Descripción"].unique().tolist()

contador = 0

for producto in productos_unicos:
    ventas_loop = ventas_productos[ventas_productos["Descripción"].isin([
                                                                        producto])]

    train, test = train_test_split(ventas_loop, test_size=0.25, random_state=42)

    FEATURES = ['dayofyear',
                'dayofweek',
                'quarter',
                'month',
                'year',
                'weekofyear',
                "diff_tiempo_venta",
                "fourier_transform",
                "trend", "seasonal"]
    
    TARGET = "Cantidad"

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]


    reg = xgb.XGBRegressor(n_estimators=1000, objective="reg:squarederror")
    # Poner quizás el eval_metric
    
    reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)
    
    test["prediction"] = reg.predict(X_test)
    y_pred = test["prediction"]
    test['Cantidad'].plot(style='b', figsize=(10, 5), label='Original')
    test['prediction'].plot(style='r', figsize=(10, 5), label='Predicción')

    # Agregar etiquetas, título, leyenda, etc.
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad')
    plt.title('Comparación de Serie Original y Predicción')
    plt.legend()

    ruta = os.path.join("plots", "forecasts", "xgboost", f"xgboost_{producto}.png")
    try:
        plt.savefig(ruta)
    except:
        ruta = os.path.join("plots", "forecasts", f"xgboost_prod_{contador}.png")
        contador += 1
        plt.savefig(ruta)
    finally:
        plt.close()
        
sys.exit()


ventas_prod = pd.read_excel(os.path.join(
    "datos", "ventas_diarias_productos_vigentes_no_outliers.xlsx"))
testeo = ventas_prod[ventas_prod["Descripción"].isin(
    ["pro plan alimento seco para adulto razas medianas 15 kg"])]
datos_forecasting = testeo[["Cantidad"]]
datos_forecasting.index = testeo["Fecha"]


datos_forecasting = apply_day_selling_differences(datos_forecasting)
datos_forecasting = apply_fourier_transform(datos_forecasting)
datos_forecasting = apply_trend_and_seasonal(datos_forecasting)
datos_forecasting = create_features(datos_forecasting)


train, test = train_test_split(
    datos_forecasting, test_size=0.25, random_state=42)


FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month', 'year', 'weekofyear', "diff_tiempo_venta", "fourier_transform",
            "trend", "seasonal"]
TARGET = "Cantidad"

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


reg = xgb.XGBRegressor(n_estimators=1000, objective="reg:squarederror",
                       eval_score=mean_absolute_percentage_error)

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)


fi = pd.DataFrame(data=reg.feature_importances_,
                  index=reg.feature_names_in_,
                  columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
# plt.close()
plt.show()
plt.close()

test["prediction"] = reg.predict(X_test)
y_pred = test["prediction"]

test['Cantidad'].plot(style='b', figsize=(10, 5), label='Original')

# Trazar la serie de predicción en rojo
test['prediction'].plot(style='r', figsize=(10, 5), label='Predicción')

# Agregar etiquetas, título, leyenda, etc.
plt.xlabel('Fecha')
plt.ylabel('Cantidad')
plt.title('Comparación de Serie Original y Predicción')
plt.legend()

# Mostrar el gráfico
plt.show()


# Construir intervalos de confianza empírico. Entrenar un mismo modelo hartas veces, para posterior
# mente tener todas las posibles estimaciones para un cierto día de esos entrenamientos y con eso
# armar el intervalo de confianza
