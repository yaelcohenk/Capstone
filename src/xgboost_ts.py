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


ventas_prod = pd.read_excel(os.path.join("datos", "ventas_diarias_productos_vigentes_no_outliers.xlsx")) 
testeo = ventas_prod[ventas_prod["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]
datos_forecasting = testeo[["Cantidad"]]
datos_forecasting.index = testeo["Fecha"]



def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear', "Cantidad", "diff_tiempo_venta",
           "fourier_transform", "trend", "seasonal"]]
    if label:
        y = df[label]
        return X, y
    return X

def apply_fourier_transform(data):
    values = data["Cantidad"].values
    fourier_transform = fft(values)
    data["fourier_transform"] = np.abs(fourier_transform)

    return data





datos_forecasting.sort_index(inplace=True)
fechas = datos_forecasting.index.to_list()
fechas = pd.DataFrame(fechas, columns=["Fecha"])

fechas["diff"] = fechas["Fecha"].diff()
fechas["diff"].fillna(pd.Timedelta(seconds=0), inplace=True)

fechas["diff"] = fechas["diff"].dt.days.astype(int)
# fechas["diff"] = fechas["diff"].days
# fechas["Fecha"] = fechas["Fecha"].days
# print(fechas.diff())
# print(fechas)

fechas = fechas["diff"].values

datos_forecasting["diff_tiempo_venta"] = fechas


# print(datos_forecasting)
datos_forecasting = apply_fourier_transform(datos_forecasting)


ventas = datos_forecasting["Cantidad"]
decomposition = sm.tsa.seasonal_decompose(ventas, period=7) # Estamos en días

trend = decomposition.trend.values
seasonal = decomposition.seasonal.values

datos_forecasting["trend"] = trend
datos_forecasting["seasonal"] = seasonal


# print(trend)


# for i in trend:
    # print(i)

# print(seasonal)
# print(datos_forecasting)
# sys.exit()





train, test = train_test_split(datos_forecasting, test_size = 0.25, random_state=42)
train = create_features(train)
test = create_features(test)



FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month', 'year', 'weekofyear', "diff_tiempo_venta", "fourier_transform",
            "trend", "seasonal"]
TARGET = "Cantidad"

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


reg = xgb.XGBRegressor(n_estimators=1000, objective="reg:squarederror", eval_score=mean_absolute_percentage_error)

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