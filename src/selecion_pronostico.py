import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from parametros import PATH_ADI_PRODUCTOS, PATH_METRICAS_CROSTON, PATH_METRICAS_HOLTWINTER, PATH_METRICAS_PROPHET, PATH_METRICAS_XGBOOST, PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS


ventas_diarias_prod_vigentes_no_outliers = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS)

productos_unicos = ventas_diarias_prod_vigentes_no_outliers["Descripción"].unique().tolist()
productos_adi = pd.read_excel(PATH_ADI_PRODUCTOS)
metricas_croston = pd.read_excel(PATH_METRICAS_CROSTON)
metricas_holt_winters = pd.read_excel(PATH_METRICAS_HOLTWINTER)
metricas_prophet = pd.read_excel(PATH_METRICAS_PROPHET)
metricas_xgboost = pd.read_excel(PATH_METRICAS_XGBOOST)

contador_holt= 0
listas = list()
metricas = ["f{metrica_croston_db}", "metrica_holt_winters_db", "metrica_prophet_db", "metrica_xgboost_db"]
for producto in productos_unicos:
    resultado_croston=0
    resultado_xgboost=0
    resultado_holt=0
    resultado_prophet=0
    
    resultados=list()
    metrica_croston_db = metricas_croston[metricas_croston["producto"].isin([producto])]
    metrica_holt_winters_db = metricas_holt_winters[metricas_holt_winters["producto"].isin([producto])]
    metrica_prophet_db = metricas_prophet[metricas_prophet["producto"].isin([producto])]
    metrica_xgboost_db = metricas_xgboost[metricas_xgboost["producto"].isin([producto])]
    metricas_producto_adi = productos_adi[productos_adi["Descripción"].isin([producto])]
    producto1 = ventas_diarias_prod_vigentes_no_outliers[ventas_diarias_prod_vigentes_no_outliers["Descripción"].isin([producto])]
    grupo = producto1["Grupo"].values[0]
    subgrupo = producto1["Subgrupo"].values[0]
    
    
  
    
    
    categoria= metricas_producto_adi["categoria"].values
    if categoria[0] =="Intermittent":
        mape_croston = float(metrica_croston_db["MAPE"].values[0])
        rmse_croston = float(metrica_croston_db["RMSE"].values[0])
        mae_croston = float(metrica_croston_db["MAE"].values[0])
        mse_croston = float(metrica_croston_db["MSE"].values[0])
        
        resultado_croston = 0 * mape_croston + 0.33* rmse_croston + 0.33* mae_croston + 0.34 * mse_croston
        resultados.append(resultado_croston)
        if float(metrica_holt_winters_db["tracking_signal"].values[0]) != -1000:
            mape_holt = float(metrica_holt_winters_db["MAPE"].values[0])
            rmse_holt = float(metrica_holt_winters_db["RMSE"].values[0])
            mae_holt = float(metrica_holt_winters_db["MAE"].values[0])
            mse_holt = float(metrica_holt_winters_db["MSE"].values[0])
            
            resultado_holt = 0 * mape_holt + 0.33* rmse_holt + 0.33* mae_holt + 0.34 * mse_holt
            resultados.append(resultado_holt)
            

        
        mape_prophet = float(metrica_prophet_db["MAPE"].values[0])
        rmse_prophet = float(metrica_prophet_db["RMSE"].values[0])
        mae_prophet = float(metrica_prophet_db["MAE"].values[0])
        mse_prophet = float(metrica_prophet_db["MSE"].values[0])
        resultado_prophet = 0 * mape_prophet + 0.33* rmse_prophet + 0.33* mae_prophet + 0.34 * mse_prophet
        resultados.append(resultado_prophet)
        
        mape_xgboost = float(metrica_xgboost_db["MAPE"].values[0])
        rmse_xgboost = float(metrica_xgboost_db["RMSE"].values[0])
        mae_xgboost = float(metrica_xgboost_db["MAE"].values[0])
        mse_xgboost = float(metrica_xgboost_db["MSE"].values[0])
        resultado_xgboost = 0 * mape_xgboost + 0.33* rmse_xgboost + 0.33* mae_xgboost + 0.34 * mse_xgboost
        resultados.append(resultado_xgboost)
        
        if resultado_croston ==0:
            if len(resultados)==4:
                listas.append([producto, grupo, subgrupo, "croston", True])
        else:
            mejor_valor= min(resultados)
            index = resultados.index(mejor_valor)
            if len(resultados)==4:
                if index == 0:
                    listas.append([producto, grupo, subgrupo, "croston", True])
                if index == 1:
                    listas.append([producto, grupo, subgrupo, "holt_winters", True])
                if index == 2:
                    listas.append([producto, grupo, subgrupo, "prophet", True])
                if index == 3:
                    listas.append([producto, grupo, subgrupo, "xgboost", True])
            else:
                if index == 0:
                    listas.append([producto, grupo, subgrupo, "croston", False])
                if index == 1:
                    listas.append([producto, grupo, subgrupo, "prophet", False])
                if index == 2:
                    listas.append([producto, grupo, subgrupo, "xgboost", False])
    else:
        mape_croston = float(metrica_croston_db["MAPE"].values[0])
        rmse_croston = float(metrica_croston_db["RMSE"].values[0])
        mae_croston = float(metrica_croston_db["MAE"].values[0])
        mse_croston = float(metrica_croston_db["MSE"].values[0])
        
        resultado_croston = 0.25 * mape_croston + 0.25* rmse_croston + 0.25* mae_croston + 0.25 * mse_croston
        
        resultados.append(resultado_croston)
        
        if float(metrica_holt_winters_db["tracking_signal"].values[0]) != -1000:
            mape_holt = float(metrica_holt_winters_db["MAPE"].values[0])
            rmse_holt = float(metrica_holt_winters_db["RMSE"].values[0])
            mae_holt = float(metrica_holt_winters_db["MAE"].values[0])
            mse_holt = float(metrica_holt_winters_db["MSE"].values[0])
            
            resultado_holt = 0.25 * mape_holt + 0.25* rmse_holt + 0.25* mae_holt + 0.25 * mse_holt
            resultados.append(resultado_holt)
        
        mape_prophet = float(metrica_prophet_db["MAPE"].values[0])
        rmse_prophet = float(metrica_prophet_db["RMSE"].values[0])
        mae_prophet = float(metrica_prophet_db["MAE"].values[0])
        mse_prophet = float(metrica_prophet_db["MSE"].values[0])
        
        resultado_prophet = 0.25 * mape_prophet + 0.25* rmse_prophet + 0.25* mae_prophet + 0.25 * mse_prophet
        
        resultados.append(resultado_prophet)
        
        mape_xgboost = float(metrica_xgboost_db["MAPE"].values[0])
        rmse_xgboost = float(metrica_xgboost_db["RMSE"].values[0])
        mae_xgboost = float(metrica_xgboost_db["MAE"].values[0])
        mse_xgboost = float(metrica_xgboost_db["MSE"].values[0])
        resultado_xgboost = 0.25 * int(mape_xgboost) + 0.25* int(rmse_xgboost) + 0.25* int(mae_xgboost) + 0.25 * int(mse_xgboost)
        
        resultados.append(resultado_xgboost)
        
        mejor_valor= min(resultados)
        index = resultados.index(mejor_valor)
        
        if len(resultados)==4:
            if index == 0:
                listas.append([producto, grupo, subgrupo, "croston",True])
            if index == 1:
                listas.append([producto, grupo, subgrupo, "holt_winters",True])
            if index == 2:
                listas.append([producto, grupo, subgrupo, "prophet",True])
            if index == 3:
                listas.append([producto, grupo, subgrupo, "xgboost",True])
        else:
            if index == 0:
                listas.append([producto, grupo, subgrupo, "croston", False])
            if index == 1:
                listas.append([producto, grupo, subgrupo, "prophet", False])
            if index == 2:
                listas.append([producto, grupo, subgrupo, "xgboost", False])
        
        


dataset = pd.DataFrame(listas, columns=["Descripción","Grupo", "Subgrupo","Modelo de pronostico", "Boolean"])
print(resultados)
dataset.to_excel(os.path.join("datos", "Selección_modelo.xlsx"), index=False)
contador = 0
predicciones_propuesta= list()
for eleccion in listas:
    print()
    
    fechas_pronostico = list()
    
    modelo = eleccion[3]
    if modelo =="holt_winters": 
        predicion_producto = pd.read_excel(os.path.join("predicciones",f"{modelo}", f"producto_{contador_holt}.xlsx"))
    else:
        predicion_producto = pd.read_excel(os.path.join("predicciones",f"{modelo}", f"producto_{contador}.xlsx"))
    pronostico_producto = predicion_producto["predicciones"].values.tolist()
    
        
    fecha_pronostico = predicion_producto["Fecha"].tolist()
        
   
    if eleccion[4]==True:
        contador_holt+=1
    l=0
    for pronostico in pronostico_producto:
        predicciones_propuesta.append([eleccion[0], fecha_pronostico[l], int(pronostico), eleccion[1], eleccion[2] ])
        l+=1
    contador+=1

data = pd.DataFrame(predicciones_propuesta, columns=["Descripción","Fecha","predicción", "Grupo", "Subgrupo "])

data.to_excel(os.path.join("predicciones", "predicciones_propuesta.xlsx"), index=False)