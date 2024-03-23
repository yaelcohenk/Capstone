import matplotlib.pyplot as plt

def graficar_seaborn(tipo_grafico, data, x, y, xlabel, ylabel, path):
    fig, ax = plt.subplots()
    grafico = tipo_grafico(data=data, x=x, y=y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    fig = grafico.get_figure()
    fig.savefig(path)

    plt.close()


