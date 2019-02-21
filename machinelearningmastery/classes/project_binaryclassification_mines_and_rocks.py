
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#import warnings to remove warning complaints
import warnings

warnings.filterwarnings("ignore",category=FutureWarning)

#This is a more urgent warning. Removing may cause code to error if python upgraded
warnings.filterwarnings("ignore",category=DeprecationWarning)

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")

class ProjectMinesAndRocks(object):

	def __init__(self):
		#assigning a class variable 'file' to instance attribute (or variable) self.filename
		self.filename = 'sonar.all-data.csv'
		self.dataset = read_csv(self.filename,header=None)	

	def runTest(self):
		print("Hello World Mines and Rocks")

	def dataExplore(self):
		dataset = self.dataset
		print("\n*****DATA EXPLORATION*****\n")
		print("Datashape:\n{}\n".format(dataset.shape))
		set_option('display.max_rows',500)
		print("Dataset datatypes:\n{}\n".format(dataset.dtypes))
		set_option('display.width',100)
		print("Dataset head(20):\n{}\n".format(dataset.head(20)))

		set_option('precision',3)
		print("Dataset describe:\n{}\n".format(dataset.describe()))
		print("Dataset Group By:\n{}\n".format(dataset.groupby(60).size()))		

	def uniPlots(self):
		dataset = self.dataset
		
		#plots in succession
		#histogram
		dataset.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
		pyplot.show()
		
		#density
		dataset.plot(kind='density',subplots=True,layout=(8,8),sharex=False,legend=False,fontsize=1)
		pyplot.show()
		
		#boxplot
		#dataset.plot(kind='box',subplots=True,layout=(8,8),sharex=False,sharey=False)
		#pyplot.show()

	def multiplots(self):
		dataset = self.dataset

		#plots in succession
		#scatter_matrix
		#print("\nPlotting ScatterMatrix: please wait...")
		#scatter_matrix(dataset)
		#pyplot.show()

		#correlation matrix
		print("\nPlotting Correlation: please wait...")
		fig = pyplot.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(dataset.corr(),vmin=-1,vmax=1,interpolation='none')
		fig.colorbar(cax)
		#ticks = numpy.arange(0,14,1)
		#ax.set_xticks(ticks)
		#ax.set_yticks(ticks)
		#ax.set_xticklabels(self.names)
		#ax.set_yticklabels(self.names)
		pyplot.show()

	def evaluateAlgorithm(self):
		array = self.dataset.values
		X = array[:,0:60]
		Y = array[:,60]
		validation_size = 0.20
		seed = 7
		X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,random_state=seed)
		
		num_folds = 10
		seed = 7 
		scoring = 'accuracy'

		models = []
		models.append(('LR',LogisticRegression()))
		models.append(('LDA',LinearDiscriminantAnalysis()))
		models.append(('KNN',KNeighborsClassifier()))
		models.append(('CART',DecisionTreeClassifier()))
		models.append(('NB',GaussianNB()))
		models.append(('SVM',SVC()))

		results = []
		names = []
		for name, model in models:
			kfold = KFold(n_splits=num_folds,random_state=seed)
			cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "NAME:{}, MEAN:{}, STD:{}".format(name, cv_results.mean(), cv_results.std())
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

		#Add Standardization
		print("\n\nStandardization")
		pipelines = []
		pipelines.append(('ScaledLR',Pipeline([('Scaler',StandardScaler()), ('LR',LogisticRegression())])))
		pipelines.append(('ScaledLDA',Pipeline([('Scaler',StandardScaler()), ('LASSO',LinearDiscriminantAnalysis())])))
		pipelines.append(('ScaledKNN',Pipeline([('Scaler',StandardScaler()), ('KNN',KNeighborsClassifier())])))
		pipelines.append(('ScaledCART',Pipeline([('Scaler',StandardScaler()), ('CART',DecisionTreeClassifier())])))
		pipelines.append(('ScaledNB',Pipeline([('Scaler',StandardScaler()), ('NB',GaussianNB())])))
		pipelines.append(('ScaledSVM',Pipeline([('Scaler',StandardScaler()), ('SVM',SVC())])))
		results = []
		names = []
		#exit()

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




	def algorithmTuning(self):
		array = self.dataset.values
		X = array[:,0:60]
		Y = array[:,60]
		validation_size = 0.20
		seed = 7
		X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,random_state=seed)
		
		num_folds = 10
		seed = 7 
		scoring = 'accuracy'

		#tuning KNN
		scaler = StandardScaler().fit(X_train)
		rescaledX = scaler.transform(X_train)
		neighbors = [1,3,5,7,9,11,13,15,17,19,21]
		param_grid = dict(n_neighbors=neighbors)
		model = KNeighborsClassifier()
		kfold = KFold(n_splits=num_folds, random_state=seed)
		grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring=scoring,cv=kfold)
		grid_result = grid.fit(rescaledX,Y_train)

		print("Best: {} using {}".format(grid_result.best_score_,grid_result.best_params_))
		means = grid_result.cv_results_['mean_test_score']
		stds = grid_result.cv_results_['std_test_score']
		params = grid_result.cv_results_['params']
		for mean, stdev, param in zip(means,stds,params):
			print("mean:{}, stdev:{}, param:{}".format(mean,stdev,param))


		#tuning SVM
		scaler = StandardScaler().fit(X_train)
		rescaledX = scaler.transform(X_train)
		c_values = [0.1,0.3,0.5,0.7,0.9,1.0,1.3,1.5,1.7,2.0]
		kernel_values = ['linear','poly','rbf','sigmoid']
		param_grid = dict(C=c_values,kernel=kernel_values)
		model = SVC()
		kfold = KFold(n_splits=num_folds,random_state=seed)
		grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
		grid_result = grid.fit(rescaledX,Y_train)

		print("Best: {} using {}".format(grid_result.best_score_,grid_result.best_params_))
		means = grid_result.cv_results_['mean_test_score']
		stds = grid_result.cv_results_['std_test_score']
		params = grid_result.cv_results_['params']
		for mean, stdev, param in zip(means,stds,params):
			print("mean:{}, stdev:{}, param:{}".format(mean,stdev,param))

	def algorithmsEnsemble(self):
		array = self.dataset.values
		X = array[:,0:60]
		Y = array[:,60]
		validation_size = 0.20
		seed = 7
		X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,random_state=seed)
		
		num_folds = 10
		seed = 7 
		scoring = 'accuracy'

		#ensembles
		ensembles = []
		ensembles.append(('AB',AdaBoostClassifier()))
		ensembles.append(('GBM',GradientBoostingClassifier()))
		ensembles.append(('RF',RandomForestClassifier()))
		ensembles.append(('ET',ExtraTreesClassifier()))

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


		#tuning SVM
		scaler = StandardScaler().fit(X_train)
		rescaledX = scaler.transform(X_train)
		model = SVC(C=1.5)
		model.fit(rescaledX,Y_train)
		rescaledValidationX=scaler.transform(X_validation)
		predictions = model.predict(rescaledValidationX)
		print(accuracy_score(Y_validation,predictions))
		print(confusion_matrix(Y_validation,predictions))
		print(classification_report(Y_validation,predictions))






