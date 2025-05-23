import pandas as pd
import os
import sys
import statsmodels.api as sm
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from scipy.fft import fft


def apply_fourier_transform(data):
    values = data["Cantidad"].values
    fourier_transform = fft(values)
    data["fourier_transform"] = np.abs(fourier_transform)

    return data


def apply_trend_and_seasonal(data, period=7):
    ventas = data["Cantidad"]
    try:
        decomposition = sm.tsa.seasonal_decompose(ventas, period=period)
    except ValueError:
        # decomposition = sm.tsa.seasonal_decompose(ventas)
        data["trend"] = None
        data["seasonal"] = None
        return data
    
    trend = decomposition.trend.values
    seasonal = decomposition.seasonal.values

    data["trend"] = trend
    data["seasonal"] = seasonal

    return data


def apply_day_selling_differences(data):
    data = data.sort_index()
    fechas = data.index.to_list()
    fechas = pd.DataFrame(fechas, columns=["Fecha"])

    fechas["diff"] = fechas["Fecha"].diff()
    fechas["diff"].fillna(pd.Timedelta(seconds=0), inplace=True)

    fechas["diff"] = fechas["diff"].dt.days.astype(int)

    fechas = fechas["diff"].values

    data["diff_tiempo_venta"] = fechas

    return data


def create_features(df):
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

    X = df[['dayofweek', 'quarter', 'month', 'year',
           'dayofyear', 'dayofmonth', 'weekofyear', "Cantidad", "diff_tiempo_venta",
            "fourier_transform", "trend", "seasonal", "Descripción"]]

    return X


def agregar_demanda_fecha_anterior(df):
    df["ventas_fecha_anterior"] = df["Cantidad"].shift(1)
    return df