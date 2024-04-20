import pandas as pd
import seaborn as sns
import os
import sys
import xgboost as xgb
import matplotlib.pyplot as plt

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


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
           'dayofyear','dayofmonth','weekofyear', "Cantidad"]]
    if label:
        y = df[label]
        return X, y
    return X


# datos = create_features(datos_forecasting)
# print(datos)


train, test = train_test_split(datos_forecasting, test_size = 0.2, random_state=42)


train = create_features(train)
# print(train)
test = create_features(test)

# print(train)
# sys.exit()

FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month', 'year', 'weekofyear']
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
# fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
# plt.close()
# plt.show()

test["prediction"] = reg.predict(X_test)


# print(X_train)

# sys.exit()
# print(test)


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
