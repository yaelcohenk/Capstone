import pandas as pd
import os
import orbit

from orbit.models import ETS
from orbit.diagnostics.plot import plot_predicted_data


from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES

ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
# ventas_productos.index = ventas_productos["Fecha"]
ventas_productos = ventas_productos[ventas_productos["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]

TEST_SIZE = 90
train_df = ventas_productos[:-TEST_SIZE]
test_df = ventas_productos[-TEST_SIZE:]

response_col="Cantidad"
date_col = "Fecha"

ets = ETS(response_col=response_col, date_col=date_col, seasonality=7, seed=2024,
          estimator="stan-mcmc", stan_mcmc_args={"show_progress": False})


ets.fit(df=train_df)
predicted_df = ets.predict(df=test_df)

_ = plot_predicted_data(train_df, predicted_df, date_col, response_col, title='Prediction with ETS')

# print(ventas_productos)