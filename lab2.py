from sklearn import neighbors
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score as accuracy
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV



def rbf_kernel(distance, h):
	return np.exp(-1/(2*h) * distance)

def epan_kernel(distance, h):
	return 3/4*(1 - distance ** 2 / h ** 2)


class KKNNClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, kernel=rbf_kernel, h=10):
		self.kernel = kernel
		self.h = h
		self.distance = lambda a, b: np.linalg.norm(a-b)

	def fit(self, x, y):
		self._train_features = x.values
		self._train_tags = y.values
		self._weight = lambda x: self.kernel(x, self.h)

	def _predict_routine(self, x):

		distances = [self.distance(x, train) for train in self._train_features]
		norm = np.linalg.norm(distances)
		weightened_dist = [self._weight(d) for d in distances]
		classes = [0 for _ in range(len(set(self._train_tags)))]

		for el, tag in zip(weightened_dist, self._train_tags):
			classes[tag] += el

		return np.argmax(classes)

	def predict(self, x):
		try:
			getattr(self, "_train_features")
		except AttributeError:
			raise RuntimeError("You must train classifer before predicting data!")
			
		return [self._predict_routine(el) for el in x.values]

	def score(self, x, y):
		predicted = self.predict(x)
		return accuracy(y, predicted)





with open('Tic_tac_toe.txt') as f:
	dataset = pd.read_csv(f, header=None)


with open('spambase/spambase.data') as f:
	dataset = pd.read_csv(f, header=None)

print(len(dataset))


for i in list(dataset):
	dataset[i] = pd.factorize(dataset[i])[0]

acc = list()
dataset = shuffle(dataset)
train_acc = list()

sizes = [100, 500, 1000]
train_size = 4000
h_width = [5, 10, 100, 1000]

for h in h_width:
	#clf = neighbors.KNeighborsClassifier(h)
	clf = KKNNClassifier(h=h, kernel=epan_kernel)
	print(clf)
	clf.fit(dataset[dataset.columns[:-1]][:train_size], 
		dataset[dataset.columns[-1]][:train_size])
	acc.append(clf.score(dataset[dataset.columns[:-1]][4000:], dataset[dataset.columns[-1]][4000:]))

print(acc)
plt.plot(acc)
plt.title('Точность на тестовой выборке в зависимости от размера обучающей, Гауссово ядро, h = 7')
plt.xlabel('Размер обучающей выборки')
plt.xticks(range(len(h_width)), h_width)
plt.ylabel('Точность классификации')
plt.show()