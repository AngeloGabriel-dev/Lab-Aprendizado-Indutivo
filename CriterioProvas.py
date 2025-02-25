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


data,meta = arff.loadarff('./CriterioProvas.arff')

attributes = meta.names()
data_value = np.asarray(data)

p1 = np.asarray(data['P1']).reshape(-1,1)
p2 = np.asarray(data['P2']).reshape(-1,1)
percFalta = np.asarray(data['PercFalta']).reshape(-1,1)
features = np.concatenate((p1, p2, percFalta),axis=1)
target = data['resultado']


Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

plt.figure(figsize=(10, 6.5))
tree.plot_tree(Arvore,feature_names=['P1', 'P2','PercFalta'],class_names=['Aprovado', 'Reprovado'],
                   filled=True, rounded=True)
plt.show()

fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore,features,target,display_labels=['Aprovado', 'Reprovado'], values_format='d', ax=ax)
plt.show()
