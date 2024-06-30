import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

datos_s_s = pd.read_excel("s_s_pronosticos_escenarios.xlsx")
datos_r_q = pd.read_excel("r_q_pronosticos_escenarios.xlsx")

datos_s_s_tcero = pd.read_excel("s_s_pronosticos_escenarios_tcero.xlsx")
datos_r_q_tcero = pd.read_excel("r_q_pronosticos_escenarios_tcero.xlsx")


datos_dif = (datos_r_q["utilidad_total"] - datos_s_s["utilidad_total"])


plt.figure(figsize=(8, 6))
plt.boxplot([datos_s_s["utilidad_total"],
             datos_s_s_tcero["utilidad_total"],
             datos_r_q["utilidad_total"],
             datos_r_q_tcero["utilidad_total"]],
            labels=["Datos (T, s, S)",
                    "Datos (T, s, S) con T = 0",
                    "Datos (T, r, Q)",
                    "Datos (T, r, Q) con T = 0"
                    ])

plt.title("Utilidades distintos escenarios para las políticas de gestión de inventario")
plt.ylabel("Utilidad total")

plt.savefig("comparacion_utilidades_tcero.png")
# plt.legend()

plt.show()
# datos

# print(datos_s_s)

sys.exit()

sns.boxplot(data=datos_s_s, x="utilidad_total")
plt.title("Utilidad distintos escenarios para política (T, s, S)")

plt.show()


# Hacer los boxplots en un mismo gráfico
# print(datos_s_s)
