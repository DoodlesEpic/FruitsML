# -*- coding: utf-8 -*-
from sklearn import tree

# Dados de treinamento
#features = [[140, "lisa"], [130, "lisa"], [150, "irregular"], [170, "irregular"]]
#labels   = ["maçã",        "maçã",        "laranja",          "laranja"]
features  = [[140, 1],      [130, 1],      [150, 0],           [170, 0]]
labels    = [0,             0,             1,                  1]

# Treinamento
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)

# Classficar
print clf.predict([[135, 1]])

# Visualização
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names=["peso", "casca_lisa"],
                        class_names=["maca", "laranja"],
                        filled=True, rounded=True,
                        impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("frutas.pdf")