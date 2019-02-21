
from pandas import read_csv

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

class ensemblePerformance(object):

	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']


	def __init__(self):
		self.name = 'name'

	@classmethod
	def baggedTrees(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		seed = 7
		kfold = KFold(n_splits=10, random_state=seed)
		cart = DecisionTreeClassifier()
		num_trees = 100
		model = BaggingClassifier(base_estimator=cart,n_estimators=num_trees,random_state=seed)
		results = cross_val_score(model,X,Y,cv=kfold)
		print(results.mean())

	@classmethod
	def randomForest(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_trees = 100
		max_features= 3
		kfold = KFold(n_splits=10, random_state=7)
		model = RandomForestClassifier(n_estimators=num_trees,max_features=max_features)
		results = cross_val_score(model,X,Y,cv=kfold)
		print(results.mean())

	@classmethod
	def extraTrees(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_trees = 100
		max_features= 3
		kfold = KFold(n_splits=10, random_state=7)
		model = ExtraTreesClassifier(n_estimators=num_trees,max_features=max_features)
		results = cross_val_score(model,X,Y,cv=kfold)
		print(results.mean())

	@classmethod
	def adaBoost(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_trees = 100
		seed = 7
		kfold = KFold(n_splits=10, random_state=7)
		model = AdaBoostClassifier(n_estimators=num_trees,random_state=seed)
		results = cross_val_score(model,X,Y,cv=kfold)
		print(results.mean())

	@classmethod
	def gradBoost(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_trees = 100
		seed = 7
		kfold = KFold(n_splits=10, random_state=7)
		model = GradientBoostingClassifier(n_estimators=num_trees,random_state=seed)
		results = cross_val_score(model,X,Y,cv=kfold)
		print(results.mean())

	@classmethod
	def votingEnsemble(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		kfold = KFold(n_splits=10, random_state=7)
		#create the sub models
		estimators = []
		model1 = LogisticRegression()
		model2 = DecisionTreeClassifier()
		model3 = SVC()
		estimators.append(('logistic',model1))
		estimators.append(('cart',model2))
		estimators.append(('svm',model3))
		#create the ensemble model
		ensemble = VotingClassifier(estimators)
		results = cross_val_score(ensemble,X,Y,cv=kfold)
		print(results.mean())




