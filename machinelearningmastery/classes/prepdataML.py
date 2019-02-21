
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

class MLDataPrepper(object):

	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

	#constructor
	def __init__(self):
		self.name = 'test'	

	@classmethod 
	def rescaleData(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values

		#separate array into inpput and output components
		X = array[:,0:8]
		y = array[:,8]

		print("\nRescaled with MinMaxScaler")
		scaler = MinMaxScaler(feature_range=(0,1))
		rescaledX = scaler.fit_transform(X)
		#summarize transformed data
		set_printoptions(precision=3)
		print(rescaledX[0:5,:])

		print("\nRescaled with StandardScaler")
		scaler = StandardScaler().fit(X)
		rescaledX = scaler.transform(X)
		#summarize transformed data
		set_printoptions(precision=3)
		print(rescaledX[0:5,:])

		print("\nRescaled with Normalizer")
		scaler = Normalizer().fit(X)
		normalizedX = scaler.transform(X)
		#summarize trasnformed data
		set_printoptions(precision=3)
		print(normalizedX[0:5,:])

		binarizer = Binarizer(threshold=0.0).fit(X)
		binaryX = binarizer.transform(X)
		#summarize transformed data
		set_printoptions(precision=3)
		print(binaryX[0:5,:])