
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit

class algEvaluator(object):

	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
	

	def __init__(self):
		self.name = 'test'

	@classmethod
	def trainTestSplit(cls):
		dataframe = read_csv(cls.filename, names=cls.names)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		test_size = 0.33
		seed = 7
		X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size,random_state=seed)
		model = LogisticRegression()
		model.fit(X_train,Y_train)
		result = model.score(X_test,Y_test)
		print("Accuracy: {}%".format(result*100.0))

	@classmethod
	def kfoldCross(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_folds = 10
		seed = 7
		kfold = KFold(n_splits=num_folds,random_state=seed)
		model = LogisticRegression()
		results = cross_val_score(model, X, Y, cv=kfold)
		print("Accuracy- mean:{}%, std:{}%".format(results.mean()*100.0,results.std()*100.0))

	@classmethod
	def leaveOneOutCross(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		num_folds = 10
		loocv = LeaveOneOut()
		model = LogisticRegression()
		results = cross_val_score(model, X, Y, cv=loocv)
		print("Accuracy- mean:{}%, std:{}%".format(results.mean()*100.0,results.std()*100.0))

	@classmethod
	def repeatedRandomTrainSplits(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		n_splits = 10
		test_size = 0.33
		seed = 7
		kfold = ShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=seed)
		model = LogisticRegression()
		results = cross_val_score(model,X,Y,cv=kfold)
		print("Accuracy- mean:{}%, std:{}%".format(results.mean()*100.0,results.std()*100.0))




