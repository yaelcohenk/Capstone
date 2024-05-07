import numpy as np

def crear_datos_RNN(n_future, n_past, x_train_scaled, y_train_scaled, dataframe):
    X_train = list()
    Y_train = list()

    for i in range(n_past, len(x_train_scaled) - n_future + 1):
        X_train.append(x_train_scaled[i - n_past:i, 0:dataframe.shape[1]])
        Y_train.append(y_train_scaled[i + n_future - 1:i + n_future, 0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    Y_train = Y_train.squeeze()
    X_train_shape = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))


    return X_train_shape, Y_train