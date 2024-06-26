import pandas as pd
import os
import numpy as np
import scipy.stats as stats
import sys


datos_s_s = pd.read_excel("s_s_pronosticos_escenarios.xlsx")
datos_r_q = pd.read_excel("r_q_pronosticos_escenarios.xlsx")


utilidad_s_s = datos_s_s["utilidad_total"].to_numpy()
utilidad_r_q = datos_r_q["utilidad_total"].to_numpy()

var_diferencia = utilidad_r_q - utilidad_s_s

media = np.mean(var_diferencia)

suma_cuadrado = sum((var_diferencia - media) ** 2)
varianza = (1 / (len(var_diferencia) - 1)) * suma_cuadrado

alpha = 0.05
percentil_t_student = stats.t.ppf(1 - alpha / 2, len(var_diferencia) - 1)
ancho = percentil_t_student * np.sqrt(varianza / len(var_diferencia))
print(f"El intervalo de confianza para la comparación de políticas es de [{media - ancho}; {media + ancho}]")

