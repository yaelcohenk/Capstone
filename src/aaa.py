import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES

# Asegúrate de importar StandardScaler
from sklearn.preprocessing import StandardScaler

# Suponiendo que tienes definido PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
# ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
# ...

ventas_productos = pd.read_excel(PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
ventas_productos = ventas_productos[ventas_productos["Descripción"].isin(
    ["pro plan alimento seco para adulto razas medianas 15 kg"])]

# Elimina la columna "Fecha" y "Descripción" si ya no las necesitas
ventas_productos.drop(["Fecha", "Descripción"], axis=1, inplace=True)

# Escala los datos
scaler = StandardScaler()
df_for_training_scaled = scaler.fit_transform(ventas_productos)

# Define n_future y n_past
n_future = 1
n_past = 7

# Formatea los datos de entrada
trainX = []
trainY = []
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, :])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

# Construye el modelo LSTM
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# Entrena el modelo
history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

# Grafica la pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

sys.exit()
