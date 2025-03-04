import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.io import arff


data,meta = arff.loadarff('./tenis.arff')

attributes = meta.names()
data_value = np.asarray(data)

Tempo = np.asarray(data['Tempo']).reshape(-1, 1)
Temperatura = np.asarray(data['Temperatura']).reshape(-1, 1)
Umidade = np.asarray(data['Umidade']).reshape(-1, 1)
Vento = np.asarray(data['Vento']).reshape(-1, 1)

features = np.concatenate((Tempo, Temperatura, Umidade, Vento),axis=1)
target = data['Partida']

Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

plt.figure(figsize=(10, 6.5))
tree.plot_tree(Arvore,feature_names=['Tempo', 'Temperatura', 'Umidade', 'Vento'],class_names=['NAO', 'SIM'],
                   filled=True, rounded=True)
plt.show()

fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore,features,target,display_labels=['NAO', 'SIM'], values_format='d', ax=ax)
plt.show()
