import pandas as pd
import os
import numpy as np


metricas_xgboost = pd.read_excel(os.path.join("datos", "metricas_xgboost.xlsx"))
metricas_prophet = pd.read_excel(os.path.join("datos", "metricas_prophet.xlsx"))
metricas_holt_winters = pd.read_excel(os.path.join("datos", "metricas_holt_winters.xlsx"))

menores_mape_xgboost_v_prophet = metricas_xgboost["MAPE"] < metricas_prophet["MAPE"]
menores_mape_xgboost_v_holt_winters = metricas_xgboost["MAPE"] < metricas_holt_winters["MAPE"]
menores_mape_prophet_v_holt_winters = metricas_prophet["MAPE"] < metricas_holt_winters["MAPE"]

menores_holt_winters_v_xgboost = metricas_xgboost["MAPE"]  > metricas_holt_winters["MAPE"]

porcentaje_xgboost_v_prophet = (sum(menores_mape_xgboost_v_prophet) / len(metricas_xgboost)) * 100
porcentaje_xgboost_v_holtwinter = (sum(menores_mape_xgboost_v_holt_winters) / len(metricas_xgboost)) * 100

porcentaje_prophet_v_holt_winters = (sum(menores_mape_prophet_v_holt_winters) / len(metricas_xgboost)) * 100

porcentaje_holt_vs_xgboost = (sum(menores_holt_winters_v_xgboost) / len(metricas_xgboost)) * 100


mape_xgboost = metricas_xgboost["MAPE"]
mape_prophet = metricas_prophet["MAPE"]
mape_holt = metricas_holt_winters["MAPE"].replace(np.inf, 100)


print(sum(mape_xgboost == mape_holt))


print(mape_holt)
