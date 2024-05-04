import pandas as pd
import os
import calplot
import matplotlib.pyplot as plt
from funciones.plotting import graficar_heatmap_datos
import seaborn as sns


porcentajes_acumulados = pd.read_excel(
    os.path.join("datos", "ganancias_total_productos.xlsx"))
porcentajes_acumulados["rango"] = range(1, len(porcentajes_acumulados) + 1)
print(porcentajes_acumulados)

fig, ax = plt.subplots(figsize=(10, 10))

sns.lineplot(data=porcentajes_acumulados, x="rango",
             y="porcentaje_marginal_acumulado")

ax.set_ylabel("Porcentaje Ingreso Marginal Acumulado")
ax.set_xlabel("Cantidad productos más vendidos (orden descendente)")

ax.axvline(x=75, ymin=0, ymax=0.76, color='r', linestyle='--', label='80% del porcentaje marginal acumulado')
ax.scatter(75, 0.8, color="red")
ax.text(75, 0.015, '75', ha='center', color="red")
ax.axhline(y=1, color='r', linestyle='--')
plt.title("Porcentaje del ingreso marginal acumulado de los productos más vendidos")
plt.legend()
plt.show()

fig.savefig("grafico_pareto.png")