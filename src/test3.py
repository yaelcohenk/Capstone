import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
datos = pd.read_excel(os.path.join("datos", "ventas_diarias_subgrupos.xlsx"))
#

# data = datos.groupby("Fecha").size().reset_index()
# print(data.value_counts())

datos = datos.sort_values(by=["Fecha"], ascending=True)
datos = datos.set_index("Fecha")

tabla_dinamica = datos.pivot_table(
    index='Fecha', columns='Subcategoría', values='Cantidad', aggfunc='sum')

print(tabla_dinamica.corr())

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(tabla_dinamica.corr(), cmap="crest")
plt.title("Matriz correlación ventas diarias en los subgrupos")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
