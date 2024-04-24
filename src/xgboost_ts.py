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
from sklearn.preprocessing import StandardScaler


ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
ventas_productos.index = ventas_productos["Fecha"]
ventas_productos.drop("Fecha", axis=1, inplace=True)
# print(ventas_productos)

productos_unicos = ventas_productos["Descripción"].unique().tolist()

contador = 0

productos_unicos = ["pro plan alimento seco para adulto razas medianas 15 kg"]

# Tengo que paralelizar esto
for producto in productos_unicos:
    ventas_loop = ventas_productos[ventas_productos["Descripción"].isin([
                                                                        producto])]

    

    # Dividir mejor en train, validación y testing
    # train, test = train_test_split(ventas_loop, test_size=0.2, random_state=42)

    # test, validation = train_test_split(test, test_size=0.1, random_state=42)

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


    X_train, y_train = ventas_loop[FEATURES], ventas_loop[TARGET]
    
    scaler_trainer = StandardScaler()
    x_train_scaled = scaler_trainer.fit_transform(np.array(X_train))

    scaler_values = StandardScaler()
    y_train_scaled = scaler_values.fit_transform(np.array(y_train).reshape(-1, 1))




    X_train, X_test, y_train, y_test = train_test_split(x_train_scaled, y_train_scaled,
                                                    test_size=0.2,
                                                    random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # X_train, y_train = train[FEATURES], train[TARGET]
    # y_train = train[TARGET]

    # X_test, y_test = test[FEATURES], test[TARGET]
    # y_test = test[TARGET]

    # X_val, y_val = validation[FEATURES], validation[TARGET]





    reg = xgb.XGBRegressor(n_estimators=1000, objective="reg:squarederror")
    # Poner quizás el eval_metric
    
    reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)
    
    # test["prediction"] = reg.predict(X_val)


    prediction = reg.predict(X_val).squeeze()

    prediction = scaler_values.inverse_transform(prediction.reshape(-1, 1)).flatten()
    y_val = scaler_values.inverse_transform(y_val.reshape(-1, 1)).flatten()


    data = pd.DataFrame({"prediction": prediction, "real": y_val})
    print(data)


    # y_pred = test["prediction"]
    data['real'].plot(style='b', figsize=(10, 5), label='Original')
    data['prediction'].plot(style='r', figsize=(10, 5), label='Predicción')

    # Agregar etiquetas, título, leyenda, etc.
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad')
    plt.title('Predicciones XGBoost: Comparación de Serie Original y Predicción')
    plt.legend()
    plt.show()

    ruta = os.path.join("plots", "forecasts", "xgboost", f"xgboost_{producto}.png")
    try:
        plt.savefig(ruta)
    
    except ValueError:
        ruta = os.path.join("plots", "forecasts", f"xgboost_prod_{contador}.png")
        contador += 1
        plt.savefig(ruta)
    
    finally:
        plt.close()


# Construir intervalos de confianza empírico. Entrenar un mismo modelo hartas veces, para posterior
# mente tener todas las posibles estimaciones para un cierto día de esos entrenamientos y con eso
# armar el intervalo de confianza
