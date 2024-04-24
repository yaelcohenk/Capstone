import pandas as pd
import xgboost as xgb
import ray

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

from parametros import (FEATURES,
                        TARGET,
                        PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)

# Cada parte lo tengo que dividir en train, test, validación

# Ver si me sirve escalar los datos antes de pasarselo al xgboost

# Preguntar si tenemos que ocupar distintas métricas y de ahí comparar

# Opiniones de los modelos de pronóstico

# Quizás aplicarle un optuna a lo del Regressor

@ray.remote
def xgboost_producto(dataframe_producto: pd.DataFrame,
                     nombre_producto: str = "",
                     random_state=42):

    print(f"[INFO]: Va a empezar entrenamiento {nombre_producto}")

    try:
        train,  test = train_test_split(dataframe_producto, test_size=0.25, random_state=random_state)
        X_train, y_train = train[FEATURES], train[TARGET]

        X_test, y_test = test[FEATURES], test[TARGET]

        # Aquí en teoría debería tener un set de validación
        reg = xgb.XGBRegressor(n_estimators=1000, objective="reg:squarederror")

        reg.fit(X_train, y_train, eval_set=[
                (X_train, y_train), (X_test, y_test)], verbose=100)


        predicciones = reg.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, predicciones)
        return mape
    
    except ValueError as e:
        print(f"Ha ocurrido un error {e}")
        return "ValueError"

    # Acá podría retornar todas las métricas de interés, el modelo de regresión, el nombre
    # etc..


if __name__ == '__main__':
    ray.init()
    ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
    ventas_productos.index = ventas_productos["Fecha"]
    ventas_productos.drop("Fecha", axis=1, inplace=True)
    productos = ventas_productos["Descripción"].unique().tolist()

    # productos = productos[:10] # Esto cambiarlo de ahí para ver toods los productos

    lista_productos =[ventas_productos[ventas_productos["Descripción"].isin([i])] for i in productos]
    
    lista_final = list(zip(lista_productos, productos))

    futures = [xgboost_producto.remote(dataframe_prod, prod) for dataframe_prod, prod in lista_final]
# 
    print(ray.get(futures))

    