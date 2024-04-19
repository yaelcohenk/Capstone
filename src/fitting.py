import pandas as pd
import sys
import os
# import torch
# import pytorch_lightning as pl
# import matplotlib.pyplot as plt
# import pytorch_forecasting as ptf
# 
# from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
# from pytorch_lightning.callbacks import EarlyStopping
# from pytorch_forecasting.metrics import SMAPE, MultivariateNormalDistributionLoss
# from pytorch_forecasting.data import NaNLabelEncoder
# from lightning.pytorch.tuner import Tuner
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pytorch_forecasting as ptf

from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss
from lightning.pytorch.tuner import Tuner


if False:
    productos_vigentes = pd.read_excel(
        os.path.join("datos", "prod_vigentes.xlsx"))
    productos_vigentes = productos_vigentes["Descripción"].to_list()

    ventas_diarias_prod = pd.read_excel(
        os.path.join("datos", "ventas_diarias_productos.xlsx"))
    ventas_diarias_prod = ventas_diarias_prod[ventas_diarias_prod["Descripción"].isin(
        productos_vigentes)]

    # Según el tutorial, la librería espera el target como float. En este caso, cantidad
    ventas_diarias_prod["Cantidad"] = pd.to_numeric(
        ventas_diarias_prod["Cantidad"], downcast="float")
    min_date = ventas_diarias_prod["Fecha"].min()
    ventas_diarias_prod["time_idx"] = ventas_diarias_prod["Fecha"].map(
        lambda current_date: (current_date - min_date).days)

    ventas_diarias_prod.to_excel("dataset_fitting.xlsx", index=False)

# sys.exit()
ventas_diarias_prod = pd.read_excel("dataset_fitting.xlsx")
ventas_diarias_prod["Cantidad"] = pd.to_numeric(ventas_diarias_prod["Cantidad"], downcast="float")

max_encoder_length = 25  # days
max_prediction_length = 20
training_cutoff = ventas_diarias_prod["time_idx"].max() - max_prediction_length

# print(data)

grupos = ventas_diarias_prod["Grupo"]
subgrupos = ventas_diarias_prod["Subgrupo"]
productos = ventas_diarias_prod["Descripción"]
# print(type(grupos), type(subgrupos))
# sys.exit()

grupos_nan = NaNLabelEncoder()
subgrupos_nan = NaNLabelEncoder()
descripcion_nan = NaNLabelEncoder()

training = TimeSeriesDataSet(ventas_diarias_prod[lambda x: x.time_idx <= training_cutoff],
                             time_idx="time_idx",
                             target="Cantidad",
                             group_ids=["Descripción"],
                             categorical_encoders={"Grupo": grupos_nan.fit(grupos),
                                                   "Subgrupo": subgrupos_nan.fit(subgrupos),
                                                   "Descripción": descripcion_nan.fit(productos)},                                                  
                             max_encoder_length=max_encoder_length,
                             max_prediction_length=max_prediction_length,
                             static_categoricals=[
                                 "Descripción", "Grupo", "Subgrupo"],
                             time_varying_unknown_reals=["Cantidad"],
                             allow_missing_timesteps=True)


# sys.exit()
validation = TimeSeriesDataSet.from_dataset(training,
                                            ventas_diarias_prod,
                                            min_prediction_idx=training_cutoff + 1)


# print(training)

# sys.exit()
batch_size = 128

train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
pl.seed_everything(42)

trainer = pl.Trainer(accelerator="cpu", gradient_clip_val=1e-1)
net = DeepAR.from_dataset(
    training,
    learning_rate=3e-2,
    hidden_size=30,
    rnn_layers=2,
    loss=MultivariateNormalDistributionLoss(rank=30),
    optimizer="Adam",
)




res = Tuner(trainer).lr_find(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    min_lr=1e-5,
    max_lr=1e0,
    early_stop_threshold=400,
)

print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
# fig.close()
net.hparams.learning_rate = res.suggestion()


# print()

# print(ventas_diarias_prod.info())


# testeo = ventas_diarias_prod[ventas_diarias_prod["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])]
# ventas_diarias_prod["Descripción"].isin(["pro plan alimento seco para adulto razas medianas 15 kg"])
# print(ventas_diarias_prod)
# print(testeo)

# print
