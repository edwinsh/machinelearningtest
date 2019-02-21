
#Save Model using Pickle
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
from sklearn.externals.joblib import dump as dump2
from sklearn.externals.joblib import load as load2


class SaveAndLoad(object):
	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']	

	def __init__(self):
		self.name = 'name'

	@classmethod
	def pickleTest(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=7)
		#Fit the model on 33%
		model = LogisticRegression()
		model.fit(X_train,Y_train)
		#save the moel to disk
		filename = 'finalized_model.sav'
		dump(model,open(filename,'wb'))

		#load the model from disk
		loaded_model = load(open(filename,'rb'))
		result = loaded_model.score(X_test,Y_test)
		print(result)

	@classmethod
	def joblibTest(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=7)
		#Fit the model on 33%
		model = LogisticRegression()
		model.fit(X_train,Y_train)
		
		#save the model to disk
		filename = 'finalized_model.sav'
		dump2(model,filename)

		#load the model from disk
		loaded_model = load2(filename)
		result = loaded_model.score(X_test,Y_test)
		print(result)


