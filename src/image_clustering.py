from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import VGG16
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
from random import randint


def extract_features(file, model):
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)
    reshaped_img = img.reshape(1, 224, 224, 3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=True)
    return features


acf_plots = list()

path = os.path.join("plots", "acf_productos_vigentes")

with os.scandir(path) as files:
    for file in files:
        if file.name.endswith(".png"):
            acf_plots.append(os.path.join(path, file.name))


# print(acf_plots)

img = load_img(acf_plots[0], target_size=(224, 224))
img = np.array(img)

# print(img.shape)

reshaped_img = img.reshape(1, 224, 224, 3)

x = preprocess_input(reshaped_img)

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

data = dict()

for acf_plot in acf_plots:
    try:
        feat = extract_features(acf_plot, model)
        data[acf_plot] = feat
    except:
        pass

filenames = np.array(list(data.keys()))

feat = np.array(list(data.values()))

feat = feat.reshape(-1, 4096)
# print(feat.shape)

pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# inertia = []
# 
# for i in range(1, 15):
    # kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10)
    # kmeans.fit(x)
    # inertia.append(kmeans.inertia_)
# 
# 
# plt.plot(range(1, 15), inertia)
# plt.xlabel("Número de Clusters")
# plt.ylabel("Inercia")
# plt.show()


kmeans = KMeans(n_clusters=3, max_iter=300, n_init=10)
kmeans.fit(x)


groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)


print(groups)

groups["0"] = groups[0]
del groups[0]

groups["1"] = groups[1]
del groups[1]

groups["2"] = groups[2]
del groups[2]

with open('data.json', 'w') as fp:
    json.dump(groups, fp)