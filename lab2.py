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
	return 1 - distance ** 2 / h ** 2


class KKNNClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, kernel=rbf_kernel, h=10):
		self.kernel = kernel
		self.h = h
		self.distance = lambda a, b: np.linalg.norm(a-b)

	def fit(self, x, y):
		self._train_features = x.values
		self._train_tags = y.values
		self._classes = [0] * len(set(self._train_tags))
		self._weight = lambda x: self.kernel(x, self.h)

	def _predict_routine(self, x):

		distances = [self.distance(x, train) for train in self._train_features]
		norm_factor = sum(distances)
		weightened_dist = [self._weight(d / norm_factor) for d in distances]
		classes = self._classes

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

sizes = [100, 500, 1000, 4000]
train_size = 4000
h_width = [5, 10, 100]

# params = {'h' : h_width, 'kernel' : [rbf_kernel] * 2}
# clf = KKNNClassifier()
# grid_search = GridSearchCV(clf, params)
# grid_search.fit(dataset[dataset.columns[-1]], dataset[dataset.columns[-1]])

	#train_acc.append(np.mean(acc))
for h in h_width:
	#clf = neighbors.KNeighborsClassifier(h)
	clf = KKNNClassifier(h=h, kernel=rbf_kernel)
	print(clf)
	clf.fit(dataset[dataset.columns[:-1]][:train_size], 
		dataset[dataset.columns[-1]][:train_size])
	acc.append(clf.score(dataset[dataset.columns[:-1]][4000:], dataset[dataset.columns[-1]][4000:]))
	#acc = cross_val_score(clf, dataset[dataset.columns[:-1]][800:], dataset[dataset.columns[-1]][800:], cv=2)
	#train_acc.append(np.mean(acc))

	#acc.append(cross_val_score(clf, dataset[dataset.columns[:-1]][800:], dataset[dataset.columns[-1]][800:], cv=2))

#print(train_acc)
print(acc)
plt.plot(acc)
plt.title('Точность на тестовой выборке в зависимости от объема обучающей')
plt.xlabel('Объем обучающей выборки')
plt.xticks(range(len(acc)), sizes)
plt.ylabel('Точность классификации')
plt.show()