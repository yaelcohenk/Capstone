import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM

datos  = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
ventas_productos = datos[datos["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]


ventas_productos = ventas_productos[["Fecha", "Cantidad"]]
ventas_productos.set_index("Fecha", inplace=True)


print(ventas_productos)


# sys.exit()

train, test = train_test_split(ventas_productos, test_size=0.2)

scaler = StandardScaler()

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

print(scaled_train[:10])

n_input = 3
n_features = 1

generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


model = Sequential()
model.add(LSTM(100, activation="relu", input_shape = (n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

model.summary()

history = model.fit(generator, epochs=10)

plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()


# print(ventas_productos)