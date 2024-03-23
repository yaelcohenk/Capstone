import pandas as pd
import os
import logging
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from funciones.plotting import plot_decision_boundaries

logging.basicConfig(level=logging.DEBUG)
plt.set_loglevel(level='warning')

if __name__ == '__main__':
    GRAFICAR = False

    datos_conjuntos = pd.read_excel(
        os.path.join("datos", "datos_conjuntos.xlsx"))

    datos_kmeans = datos_conjuntos[[
        "date", "quantity", "total (CLP)", "client_id", "description", "description_2", "group_description"]]

    logging.debug(datos_kmeans)

    datos_one_hot_encoded = pd.get_dummies(datos_kmeans)
    scaler = MinMaxScaler()

    columns = datos_one_hot_encoded.columns
    columns = [i for i in columns if i != "date"]

    x = datos_one_hot_encoded[columns]
    x = x.dropna()
    x = scaler.fit_transform(x)
    x = pd.DataFrame(x, columns=[columns])

    pca = PCA(n_components=2)
    pca.fit(x)

    PCA_ds = pd.DataFrame(pca.transform(x), columns=["col1", "col2"])

    if GRAFICAR:
        fig, ax = plt.subplots()
        ax.scatter(PCA_ds["col1"], PCA_ds["col2"], marker="o", c="maroon")
        ax.set_title("Una proyección 2D de los datos en la dimensión reducida")
        plt.show()

    if GRAFICAR:
        inertia = []

        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10)
            kmeans.fit(PCA_ds)
            inertia.append(kmeans.inertia_)

        plt.plot(range(1, 11), inertia)
        plt.xlabel("Número de Clusters")
        plt.ylabel("Inercia")
        plt.show()

    # Aquí podrían ser 3 o 4. Ver cual nos permite sacar más info

    kmeans = KMeans(n_clusters=4, max_iter=300, n_init=10)
    kmeans.fit(PCA_ds)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]

    if GRAFICAR:
        fig, ax = plt.subplots()
        ax.scatter(centroids_x, centroids_y, c="red", marker="x", s=150)
        ax.scatter(x, y, c="maroon", marker="o")
        ax.set_title("A 2D Projection Of Data In The Reduced Dimension")
        plt.show()

    plt.figure(figsize=(8, 4))
    plot_decision_boundaries(kmeans, PCA_ds.to_numpy())
    # plt.show()
    plt.savefig(os.path.join("plots", "eda_conjunto", "kmeans.png"))

    plt.close()

    # datos_ohe_no_nulos = datos_one_hot_encoded.dropna()
    # datos_ohe_no_nulos["label_cluster"] = labels

    datos_kmeans_labeled = datos_kmeans.dropna()
    datos_kmeans_labeled["label_cluster"] = labels

    datos_kmeans_labeled.to_excel(os.path.join(
        "datos", "datos_kmeans_labeled.xlsx"), index=False)
