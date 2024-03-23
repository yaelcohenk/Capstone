import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def graficar_seaborn(grafico_seaborn, xlabel="", ylabel="", title="", path=None, size_x=8, size_y=8):
    plt.figure(figsize=(size_x, size_y))
    fig, ax = plt.subplots()
    grafico = grafico_seaborn
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig = grafico.get_figure()
    fig.savefig(path)
    plt.close()


def concatenar(lista):
    return pd.concat(lista)

def graficar_ganancias(data, x, y, xlabel, ylabel, title, path):
    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(data=data, x=x, y=y)

    # Rotar etiquetas del eje x
    barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()



# https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
