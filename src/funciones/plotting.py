import matplotlib.pyplot as plt


def graficar_seaborn(grafico_seaborn, xlabel, ylabel, title, path, size_x=8, size_y=8):
    plt.figure(figsize=(size_x, size_y))
    fig, ax = plt.subplots()
    grafico = grafico_seaborn
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    fig = grafico.get_figure()
    fig.savefig(path)

    plt.close()
