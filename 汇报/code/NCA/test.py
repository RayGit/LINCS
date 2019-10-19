import csv

from NCA.nca import *
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris, load_wine, load_digits, make_circles
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from os.path import join

# with open(join('E:\\Master\\LINCS\\汇报\\code\\NCA\\data', 'wine.csv')) as csv_file:
#     data_file = csv.reader(csv_file)
#     n_samples = 178
#     n_features = 13
#     X = np.empty((n_samples, n_features))
#     y = np.empty((n_samples,), dtype=np.int)
#     i = 0
#     for ir in data_file:
#         X[i] = np.asarray(ir[1:], dtype=np.float64)
#         y[i] = np.asarray(ir[0], dtype=np.int)
#         i = i + 1
# print('X: \n', X)
# print('y: \n', y)

iris_data = load_iris()
X = iris_data['data']
# print('X: \n', X)
y = iris_data['target']
# print('y: \n', y)

# wine_data = load_wine()
# X = wine_data['data']
# # print('X: \n', X)
# y = wine_data['target']
# # print('y: \n', y)

# wine_data = load_digits(n_class=5)
# X = wine_data['data']
# print('X: \n', X)
# y = wine_data['target']
# print('y: \n', y)

# X, y = make_circles(n_samples=100, shuffle=True, noise=0.05, random_state=None, factor=0.6)
# print('X: \n', X)
# print('y: \n', y)

# plt.figure(figsize=(8, 6))
# plt.clf()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
# plt.title('RAW')
# plt.xticks(())
# plt.yticks(())
# plt.show()


nca = NCA(n_components=2, max_iter=1000, verbose=True)
# nca = PCA()
nca.fit(X, y)
tx= nca.transform(X)
plt.figure(figsize=(8, 6))
plt.clf()
plt.scatter(tx[:, 0], tx[:, 1], c=y, cmap=plt.cm.Paired)
plt.title('NCA')
plt.xticks(())
plt.yticks(())
plt.show()

pca = PCA(n_components=2)
pca.fit(X, y)
tx= pca.transform(X)
plt.figure(figsize=(8, 6))
plt.clf()
plt.scatter(tx[:, 0], tx[:, 1], c=y, cmap=plt.cm.Paired)
plt.title('PCA')
plt.xticks(())
plt.yticks(())
plt.show()

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)
tx= lda.transform(X)
plt.figure(figsize=(8, 6))
plt.clf()
plt.scatter(tx[:, 0], tx[:, 1], c=y, cmap=plt.cm.Paired)
plt.title('LDA')
plt.xticks(())
plt.yticks(())
plt.show()