import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from parametros import PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES
from sklearn.model_selection import train_test_split

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError


ventas_productos = pd.read_excel(
    PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES)
ventas_productos.index = ventas_productos["Fecha"]
ventas_productos =ventas_productos[ventas_productos["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]
ventas_productos.drop("Descripción", axis=1, inplace=True)
ventas_productos.drop("Fecha", axis=1, inplace=True)



# print(ventas_productos.shape)


# TENGO QUE CONSIDERAR UN SOLO PRODUCTO

ventas_productos.dropna(inplace=True)
# print(ventas_productos.shape)

columnas_covariables = [
    columna for columna in ventas_productos.columns if columna != "Cantidad"]

x_train_covariates = ventas_productos[columnas_covariables]
y_train_value = ventas_productos[["Cantidad"]]


print(f"INFO: {y_train_value.shape}")


scaler_trainer = StandardScaler()
x_train_scaled = scaler_trainer.fit_transform(x_train_covariates)

scaler_values = StandardScaler()
y_train_scaled = scaler_values.fit_transform(y_train_value)

X_train = []
Y_train = []

# Number of days we want to look into the future based on the past days.
n_future = 1
n_past = 7  # Number of past days we want to use to predict the future.

# Reformat input data into a shape: (n_samples x timesteps x n_features)
# In my example, my df_for_training_scaled has a shape (12823, 5)
# 12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(x_train_scaled) - n_future + 1):
    X_train.append(x_train_scaled[i - n_past:i, 0:ventas_productos.shape[1]])
    Y_train.append(y_train_scaled[i + n_future - 1:i + n_future, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

Y_train = Y_train.squeeze()

X_train_shape = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))


X_train, X_test, y_train, y_test = train_test_split(X_train_shape, Y_train,
                                                    test_size=0.2,
                                                    random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Tengo que dividir el x_train -> set entrenamiento, set testeo, set validación


# model = keras.models.Sequential()
# model.add(keras.layers.LSTM(
    # units=50, activation="relu", return_sequences=True, input_shape=(7, 11)))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.LSTM(units=50, return_sequences=True, activation="relu"))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(1))



model = keras.models.Sequential([
    keras.layers.Conv1D(filters=40, kernel_size=6, activation="relu", input_shape=(7, 11)),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20),
    keras.layers.Dense(1)
])


# Este funciona
# model = keras.models.Sequential([
    # keras.layers.GRU(20, return_sequences=True, input_shape=(7, 11)),
    # keras.layers.GRU(20, return_sequences=True),
    # keras.layers.GRU(20),
    # keras.layers.Dense(1)
# ])



model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

predictions = model.predict(X_test)

# predictions = model.evaluate(X_val, y_val)

# print(f"Las predicciones son {predictions}")

predictions = predictions.squeeze()
print(predictions.shape, y_test.shape)


predictions = scaler_values.inverse_transform(predictions.reshape(-1, 1)).flatten()
real = scaler_values.inverse_transform(y_test.reshape(-1, 1)).flatten()

# print(y_train_scaled.shape)
data = pd.DataFrame({"predictions": predictions, "real": real})

print(data)


plt.plot(data["predictions"], color="red", label="Predicción")
plt.plot(data["real"], color="blue", label="Valor Real")
plt.legend()
plt.show()



# print(predictions.shape)
# print()

# print(predictions)

# print(y_test)


sys.exit()


print(f"El tamaño de las predicciones son {predictions.shape}")

# y_flatten_val = predictions.flatten()
# print(y_val.shape, y_flatten_val.shape)
y_pred = predictions[:, 0].squeeze()

y_val = y_val.reshape(-1 , 1)
y_pred = y_pred.reshape(-1, 1)

# print(y_val.shape, y_pred.shape)

# print(y_val, y_pred)

y_val_inv_transform = scaler_values.inverse_transform(y_val)
y_pred_inv_transform = scaler_values.inverse_transform(y_pred)


print(y_val_inv_transform, y_pred_inv_transform)

plt.plot(y_val_inv_transform, color = 'red', label = 'Real Stock Price')
plt.plot(y_pred_inv_transform, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



sys.exit()
predicciones_inv_trans = scaler_values.inverse_transform(predictions)
y_validation_inv = scaler_values.inverse_transform(y_val)

print(predicciones_inv_trans.shape)
print(y_validation_inv.shape)


