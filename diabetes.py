# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# import matplotlib.pyplot as plt

# data = load_iris("./CriterioProvas.arff")
# features = data.data
# target = data.target

# Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

# plt.figure(figsize=(10, 6.5))
# tree.plot_tree(Arvore, feature_names=data.feature_names, class_names=data.target_names,
#                filled=True, rounded=True)

# plt.show()

# # só lembrando: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# Amostra = [[1.9, 3.4, 0.2]]
# class_Amostra = Arvore.predict(Amostra)

import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff


data,meta = arff.loadarff('./tenis.arff')

attributes = meta.names()
data_value = np.asarray(data)


# preg = np.asarray(data['preg']).reshape(-1, 1)
# plas = np.asarray(data['plas']).reshape(-1, 1)
# pres = np.asarray(data['pres']).reshape(-1, 1)
# skin = np.asarray(data['skin']).reshape(-1, 1)
# insu = np.asarray(data['insu']).reshape(-1, 1)
# mass = np.asarray(data['mass']).reshape(-1, 1)
# pedi = np.asarray(data['pedi']).reshape(-1, 1)
# age = np.asarray(data['age']).reshape(-1, 1)

Dia = np.asarray(data['Dia']).reshape(-1, 1)
Tempo = np.asarray(data['Tempo']).reshape(-1, 1)
Temperatura = np.asarray(data['Temperatura']).reshape(-1, 1)
Umidade = np.asarray(data['Umidade']).reshape(-1, 1)
Vento = np.asarray(data['Vento']).reshape(-1, 1)
Partida = np.asarray(data['Partida']).reshape(-1, 1)

features = np.concatenate((Dia, Tempo, Temperatura, Umidade, Vento, Partida),axis=1)
target = data['Partida']


Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

plt.figure(figsize=(10, 6.5))
tree.plot_tree(Arvore,feature_names=['Dia', 'Tempo', 'Temperatura', 'Umidade', 'Vento', 'Partida'],class_names=['SIM', 'NAO'],
                   filled=True, rounded=True)
plt.show()

fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore,features,target,display_labels=['SIM', 'NAO'], values_format='d', ax=ax)
plt.show()
