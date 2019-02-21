from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class PerformanceMetrics(object):

	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

	filename2 = 'housing.csv'
	names2 = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

	def __init__(self):
		self.name = 'test'

	@classmethod
	def classificationAccuracy(cls):
		dataframe = read_csv(cls.filename, names=cls.names)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		kfold = KFold(n_splits=10,random_state=7)
		model = LogisticRegression()
		scoring = 'accuracy'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
		print("Accuracy- mean:{}%, std:{}%".format(results.mean(),results.std()))

	@classmethod
	def logarithmicLoss(cls):
		dataframe = read_csv(cls.filename, names=cls.names)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		kfold = KFold(n_splits=10,random_state=7)
		model = LogisticRegression()		
		scoring = 'neg_log_loss'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
		print("Accuracy- mean:{}%, std:{}%".format(results.mean(),results.std()))

	@classmethod
	def areaUnderCurve(cls):
		dataframe = read_csv(cls.filename, names=cls.names)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		kfold = KFold(n_splits=10,random_state=7)
		model = LogisticRegression()		
		scoring = 'roc_auc'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
		print("Accuracy- mean:{}%, std:{}%".format(results.mean(),results.std()))

	@classmethod
	def confusionMatrix(cls):
		dataframe = read_csv(cls.filename, names=cls.names)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		test_size = 0.33
		seed = 7
		X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size,random_state=seed)
		model = LogisticRegression()
		model.fit(X_train,Y_train)
		predicted = model.predict(X_test)
		matrix = confusion_matrix(Y_test,predicted)
		print(matrix)

	@classmethod
	def classificationReport(cls):
		dataframe = read_csv(cls.filename, names=cls.names)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		test_size = 0.33
		seed = 7
		X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size,random_state=seed)
		model = LogisticRegression()
		model.fit(X_train,Y_train)
		predicted = model.predict(X_test)		
		report = classification_report(Y_test,predicted)
		print(report)		

	@classmethod
	def meanAbsoluteError(cls):
		dataframe = read_csv(cls.filename2, names=cls.names2)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10,random_state=7)
		model = LinearRegression()
		scoring = 'neg_mean_absolute_error'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
		print("MAE- mean:{}%, std:{}%".format(results.mean(),results.std()))

	@classmethod
	def meanSquaredError(cls):
		dataframe = read_csv(cls.filename2,names=cls.names2)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10,random_state=7)
		model = LinearRegression()
		scoring = 'neg_mean_absolute_error'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)		
		print("MSE- mean:{}%, std:{}%".format(results.mean(),results.std()))

	@classmethod
	def rSquared(cls):
		dataframe = read_csv(cls.filename2,names=cls.names2)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10,random_state=7)
		model = LinearRegression()
		scoring = 'r2'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)		
		print("R^2- mean:{}%, std:{}%".format(results.mean(),results.std()))		










