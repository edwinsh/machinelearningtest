
import numpy
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

class AlgorithmTuner(object):
	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

	def __init__(self):
		self.name = 'name'

	@classmethod
	def gridSearch(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		
		alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
		param_grid = dict(alpha=alphas)
		model = Ridge()
		grid = GridSearchCV(estimator=model,param_grid=param_grid)
		grid.fit(X,Y)

		print(grid.best_score_)
		print(grid.best_estimator_.alpha)

	@classmethod
	def randomSearch(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		
		alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
		param_grid = {'alpha':uniform()}
		model = Ridge()
		grid = RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=100,random_state=7)
		grid.fit(X,Y)

		print(grid.best_score_)
		print(grid.best_estimator_.alpha)