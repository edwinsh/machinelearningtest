
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#linear classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#nonlinear classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class SpotCheckClassification(object):

	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

	def __init__(self):
		self.name = "test"

	@classmethod
	def logisticRegression(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_folds = 10
		kfold = KFold(n_splits=10,random_state=7)
		model = LogisticRegression()
		results = cross_val_score(model, X, Y, cv=kfold)
		print(results.mean())

	@classmethod
	def linearDiscriminantAnalysis(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_folds = 10
		kfold = KFold(n_splits=10,random_state=7)
		model = LinearDiscriminantAnalysis()
		results = cross_val_score(model, X, Y, cv=kfold)
		print(results.mean())

	@classmethod
	def kNearestNeighbors(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_folds = 10
		kfold = KFold(n_splits=10,random_state=7)
		model = KNeighborsClassifier()
		results = cross_val_score(model, X, Y, cv=kfold)
		print(results.mean())		

	@classmethod
	def naiveBayesClass(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_folds = 10
		kfold = KFold(n_splits=10,random_state=7)
		model = GaussianNB()
		results = cross_val_score(model, X, Y, cv=kfold)
		print(results.mean())	

	@classmethod
	def decisionTree(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_folds = 10
		kfold = KFold(n_splits=10,random_state=7)
		model = DecisionTreeClassifier()
		results = cross_val_score(model, X, Y, cv=kfold)
		print(results.mean())

	@classmethod
	def supportVectorMachine(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_folds = 10
		kfold = KFold(n_splits=10,random_state=7)
		model = SVC()
		results = cross_val_score(model, X, Y, cv=kfold)
		print(results.mean())


