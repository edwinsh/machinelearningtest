import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class TestClass(object):

	def __init__(self):
		self.testname = 'testname'

	@classmethod
	def numPanTest(cls):
		myarray = numpy.array([[1,2,3],[4,5,6]])
		rownames = ['a','b']
		colnames = ['one','two','three']
		mydataframe = pandas.DataFrame(myarray,index=rownames,columns=colnames)
		print(mydataframe)

	@classmethod 
	def csvLoadTest(cls):
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
		names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
		data = pandas.read_csv(url, names=names)
		print(data.shape)

	@classmethod 
	def descriptiveStats(cls):
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
		names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
		data = pandas.read_csv(url, names=names)
		description = data.describe()
		print(description)

	#decorator
	@classmethod
	def scatterMatrix(cls):
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
		names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
		data = pandas.read_csv(url, names=names)
		scatter_matrix(data)
		plt.show()

	@classmethod
	def preprocessing(cls):
		# Standardize data (0 mean, 1 stdev)
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
		names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
		dataframe = pandas.read_csv(url, names=names)
		array = dataframe.values
		# separate array into input and output components
		X = array[:,0:8]
		Y = array[:,8]
		scaler = StandardScaler().fit(X)
		rescaledX = scaler.transform(X)
		# summarize transformed data
		numpy.set_printoptions(precision=3)
		print(rescaledX[0:5,:])

	@classmethod
	def algorithmResampling(cls):
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
		names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
		dataframe = read_csv(url, names=names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		kfold = KFold(n_splits=10, random_state=7)
		model = LogisticRegression()
		results = cross_val_score(model, X, Y, cv=kfold)
		#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
		print("Results.mean {}".format(results.mean()))
		print("Results.std {}".format(results.std()))
		#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

	@classmethod
	def algorithmEvaluationMetrics(cls):
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
		names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
		dataframe = read_csv(url, names=names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		kfold = KFold(n_splits=10, random_state=7)
		model = LogisticRegression()
		scoring = 'neg_log_loss'
		results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
		print("Logloss: {} ({})".format(results.mean(), results.std()))

	@classmethod
	def knnRegression(cls):
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
		names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
		dataframe = read_csv(url, delim_whitespace=True, names=names)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10, random_state=7)
		model = KNeighborsRegressor()
		scoring = 'neg_mean_squared_error'
		results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
		print(results.mean())

	@classmethod
	def modelComparison(cls):
		# load dataset
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
		names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
		dataframe = read_csv(url, names=names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		# prepare models
		models = []
		models.append(('LR', LogisticRegression()))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		# evaluate each model in turn
		results = []
		names = []
		scoring = 'accuracy'
		for name, model in models:
			kfold = KFold(n_splits=10, random_state=7)
			cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
			print(msg)
	
	@classmethod
	def improveAccuracyWithAlgTuning(cls):
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
		names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
		dataframe = read_csv(url, names=names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
		param_grid = dict(alpha=alphas)
		model = Ridge()
		grid = GridSearchCV(estimator=model, param_grid=param_grid)
		grid.fit(X, Y)
		print(grid.best_score_)
		print(grid.best_estimator_.alpha)

	@classmethod
	def improveAccuracyWithAlgEnsemblePredictions(cls):
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
		names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
		dataframe = read_csv(url, names=names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_trees = 100
		max_features = 3
		kfold = KFold(n_splits=10, random_state=7)
		model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
		results = cross_val_score(model, X, Y, cv=kfold)
		print(results.mean())

