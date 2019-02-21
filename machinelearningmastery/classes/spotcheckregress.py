from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

class SpotCheckRegression(object):
	#class variables
	filename2 = 'housing.csv'
	names2 = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

	def __init__(self):
		self.name = 'test'	

	@classmethod
	def linearRegression(cls):
		dataframe = read_csv(cls.filename2,names=cls.names2)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10,random_state=7)
		model = LinearRegression()
		scoring = 'neg_mean_squared_error'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)		
		print(results.mean())

	@classmethod
	def ridgeRegression(cls):
		dataframe = read_csv(cls.filename2,names=cls.names2)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10,random_state=7)
		model = Ridge()
		scoring = 'neg_mean_squared_error'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)		
		print(results.mean())

	@classmethod
	def lassoRegression(cls):
		dataframe = read_csv(cls.filename2,names=cls.names2)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10,random_state=7)
		model = Lasso()
		scoring = 'neg_mean_squared_error'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)		
		print(results.mean())	

	@classmethod
	def elasticNetRegression(cls):
		dataframe = read_csv(cls.filename2,names=cls.names2)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10,random_state=7)
		model = ElasticNet()
		scoring = 'neg_mean_squared_error'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)		
		print(results.mean())	

	@classmethod
	def kneighborsRegression(cls):
		dataframe = read_csv(cls.filename2,names=cls.names2)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10,random_state=7)
		model = KNeighborsRegressor()
		scoring = 'neg_mean_squared_error'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)		
		print(results.mean())	

	@classmethod
	def decisionTreeRegression(cls):
		dataframe = read_csv(cls.filename2,names=cls.names2)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10,random_state=7)
		model = DecisionTreeRegressor()
		scoring = 'neg_mean_squared_error'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)		
		print(results.mean())			

	@classmethod
	def svrRegression(cls):
		dataframe = read_csv(cls.filename2,names=cls.names2)
		#print(dataframe)
		array = dataframe.values
		X = array[:,0:13]
		Y = array[:,13]
		kfold = KFold(n_splits=10,random_state=7)
		model = SVR()
		scoring = 'neg_mean_squared_error'
		results = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)		
		print(results.mean())	
