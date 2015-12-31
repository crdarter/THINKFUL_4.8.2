#kmeans

import numpy as np
import pylab as pl
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn import cross_validation


df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()


# split data table into data X and class labels y

X = df.ix[:,0:4].values
y = df.ix[:,4].values

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)

kf = KFold(10,n_folds=5)
for train, test in kf:
	print("%s %s" % (train, test))

	X = X_std
	y = Y_sklearn
	X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
	reduced_data = sklearn_pca.fit_transform(X_train)
	kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
	kmeans.fit(reduced_data)

	
n_samples = Y_sklearn.shape[0]
cv = cross_validation.ShuffleSplit(n_samples, n_iter=5,
	test_size=0.3, random_state=0)

print "The cross validation scores is", cross_validation.cross_val_score(kmeans, Y_sklearn, df['class'], cv=cv)

print "The mean of the cross validation scores is", np.mean(cross_validation.cross_val_score(kmeans, Y_sklearn, df['class'], cv=cv))

print "The standard deviation of the cross validation scores is", np.std(cross_validation.cross_val_score(kmeans, Y_sklearn, df['class'], cv=cv))