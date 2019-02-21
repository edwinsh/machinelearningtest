
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

#import warnings to remove warning complaints
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

#This is a more urgent warning. Removing may cause code to error if python upgraded
warnings.filterwarnings("ignore",category=DeprecationWarning)

"""
IN CONTRAST TO PREVIOUS EXERCISES, FOR PRACTICE
USING OBJECT ORIENTED PROGRAMMING STRUCTURE TO 
CALL METHODS OF AN INSTANTIATED OBJECT
"""

class ProjectBostonTest(object):

	file = 'housing.csv'

	def __init__(self):
		#assigning a class variable 'file' to instance attribute (or variable) self.filename
		self.filename = ProjectBostonTest.file
		self.names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
		self.dataset = read_csv(self.filename,names=self.names)

	def runTest(self):
		print("Hello World Boston Proj")

	def dataExplore(self):
		dataset = self.dataset
		print("\n*****DATA EXPLORATION*****\n")
		print("Datashape:\n{}\n".format(dataset.shape))
		print("Dataset datatypes:\n{}\n".format(dataset.dtypes))
		print("Dataset head(20):\n{}\n".format(dataset.head(20)))
		set_option('precision',1)
		print("Dataset describe:\n{}\n".format(dataset.describe()))
		set_option('precision',2)
		print("Dataset describe:\n{}\n".format(dataset.corr(method='pearson')))

	def uniPlots(self):
		dataset = self.dataset
		
		#plots in succession
		#histogram
		dataset.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
		pyplot.show()
		#density
		dataset.plot(kind='density',subplots=True,layout=(4,4),sharex=False,legend=False,fontsize=1)
		pyplot.show()
		#boxplot
		dataset.plot(kind='box',subplots=True,layout=(4,4),sharex=False,sharey=False)
		pyplot.show()

	def multiplots(self):
		dataset = self.dataset

		#plots in succession
		#scatter_matrix
		print("\nPlotting ScatterMatrix: please wait...")
		scatter_matrix(dataset)
		pyplot.show()

		#correlation matrix
		print("\nPlotting Correlation: please wait...")
		fig = pyplot.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(dataset.corr(),vmin=-1,vmax=1,interpolation='none')
		fig.colorbar(cax)
		ticks = numpy.arange(0,14,1)
		ax.set_xticks(ticks)
		ax.set_yticks(ticks)
		ax.set_xticklabels(self.names)
		ax.set_yticklabels(self.names)
		pyplot.show()

	def evaluateAlgorithm(self):
		array = self.dataset.values
		X = array[:,0:13]
		Y = array[:,13]
		validation_size = 0.20
		seed = 7
		X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,random_state=seed)
		num_folds = 10
		seed = 7 
		scoring = 'neg_mean_squared_error'

		models = []
		models.append(('LR',LinearRegression()))
		models.append(('LASSO',Lasso()))
		models.append(('EN',ElasticNet()))
		models.append(('KNN',KNeighborsRegressor()))
		models.append(('CART',DecisionTreeRegressor()))
		models.append(('SVR',SVR()))

		print("\n\nBaseEvaluation")
		results = []
		names = []
		for name, model in models:
			kfold = KFold(n_splits=num_folds,random_state=seed)
			cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "Name: {}, Mean:{}, STD:{}".format(name, cv_results.mean(),cv_results.std())
			print(msg)

		#render boxplots
		renderPlot = False
		if (renderPlot == True): 
			#Compare Algorithms
			fig = pyplot.figure()
			fig.suptitle('Algorithm Comparison')
			ax = fig.add_subplot(111)
			pyplot.boxplot(results)
			ax.set_xticklabels(names)
			pyplot.show()


		#sample pipeline structure
		#estimators = []
		#estimators.append(('standardize',StandardScaler()))
		#estimators.append(('lda',LinearDiscriminantAnalysis()))
		#model = Pipeline(estimators)

		#Add Standardization
		print("\n\nStandardization")
		pipelines = []
		pipelines.append(('ScaledLR',Pipeline([('Scaler',StandardScaler()), ('LR',LinearRegression())])))
		pipelines.append(('ScaledLASSO',Pipeline([('Scaler',StandardScaler()), ('LASSO',Lasso())])))
		pipelines.append(('ScaledEN',Pipeline([('Scaler',StandardScaler()), ('EN',ElasticNet())])))
		pipelines.append(('ScaledKNN',Pipeline([('Scaler',StandardScaler()), ('KNN',KNeighborsRegressor())])))
		pipelines.append(('ScaledCART',Pipeline([('Scaler',StandardScaler()), ('CART',DecisionTreeRegressor())])))
		pipelines.append(('ScaledSVR',Pipeline([('Scaler',StandardScaler()), ('SVR',SVR())])))
		results = []
		names = []
		for name, model in pipelines:
			kfold = KFold(n_splits=num_folds,random_state=seed)
			cv_results = cross_val_score(model, X_train,Y_train,cv=kfold,scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "Name: {}, Mean:{}, STD:{}".format(name, cv_results.mean(),cv_results.std())
			print(msg)

		#render boxplots
		renderPlot = False
		if (renderPlot == True): 
			#Compare Algorithms
			fig = pyplot.figure()
			fig.suptitle('Algorithm Comparison')
			ax = fig.add_subplot(111)
			pyplot.boxplot(results)
			ax.set_xticklabels(names)
			pyplot.show()

		print("\n\nKNN Algorithmtuning")
		#KNN with algorithm tuning
		scaler = StandardScaler().fit(X_train)
		rescaledX = scaler.transform(X_train)
		k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
		param_grid = dict(n_neighbors=k_values)
		model = KNeighborsRegressor()
		kfold = KFold(n_splits=num_folds,random_state=seed)
		grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
		grid_result = grid.fit(rescaledX,Y_train)

		print("Best: {} using {}".format(grid_result.best_score_,grid_result.best_params_))
		means = grid_result.cv_results_['mean_test_score']
		stds = grid_result.cv_results_['std_test_score']
		params = grid_result.cv_results_['params']
		for mean, stdev, param in zip(means,stds,params):
			print("mean:{}, stdev:{}, param:{}".format(mean,stdev,param))


	def evaluateAlgorithmEnsemble(self):
		array = self.dataset.values
		X = array[:,0:13]
		Y = array[:,13]
		validation_size = 0.20
		seed = 7
		X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,random_state=seed)
		num_folds = 10
		seed = 7 
		scoring = 'neg_mean_squared_error'

		#Ensemble 
		ensembles = []
		ensembles.append(('ScaledAB', Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostRegressor())])))
		ensembles.append(('ScaledGBM', Pipeline([('Scaler',StandardScaler()),('GBM',GradientBoostingRegressor())])))
		ensembles.append(('ScaledRF', Pipeline([('Scaler',StandardScaler()),('RF',RandomForestRegressor())])))
		ensembles.append(('ScaledET', Pipeline([('Scaler',StandardScaler()),('ET',ExtraTreesRegressor())])))

		results = []
		names = []
		for name, model in ensembles:
			kfold = KFold(n_splits=num_folds,random_state=seed)
			cv_results = cross_val_score(model, X_train,Y_train,cv=kfold,scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "Name: {}, Mean:{}, STD:{}".format(name, cv_results.mean(),cv_results.std())
			print(msg)

		#render boxplots
		renderPlot = False
		if (renderPlot == True): 
			#Compare Algorithms
			fig = pyplot.figure()
			fig.suptitle('Scaled Ensemble Algorithm Comparison')
			ax = fig.add_subplot(111)
			pyplot.boxplot(results)
			ax.set_xticklabels(names)
			pyplot.show()

		#Tune scaled GBM
		scaler = StandardScaler().fit(X_train)
		rescaledX = scaler.transform(X_train)
		param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
		model = GradientBoostingRegressor(random_state=seed)
		kfold = KFold(n_splits=num_folds,random_state=seed)
		grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
		grid_result = grid.fit(rescaledX,Y_train)

		print("Best: {} using {}".format(grid_result.best_score_,grid_result.best_params_))
		means = grid_result.cv_results_['mean_test_score']
		stds = grid_result.cv_results_['std_test_score']
		params = grid_result.cv_results_['params']
		for mean, stdev, param in zip(means,stds,params):
			print("mean:{}, stdev:{}, param:{}".format(mean,stdev,param))

		#Finalized GradientBoosting Regressor Model
		#Tune scaled GBM
		scaler = StandardScaler().fit(X_train)
		rescaledX = scaler.transform(X_train)
		model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
		model.fit(rescaledX,Y_train)
		rescaledValidationX=scaler.transform(X_validation)
		predictions = model.predict(rescaledValidationX)
		print(mean_squared_error(Y_validation,predictions))




