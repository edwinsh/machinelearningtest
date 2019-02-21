
from pandas import read_csv
from pandas import set_option

class Understander(object):

	#class variabels
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

	def __init__(self):
		self.name = 'test'

	#decorator
	@classmethod
	def utest(cls):
		print("utest tester")

	@classmethod
	def dataPeek(cls):
		data = read_csv(cls.filename,names=cls.names)
		peek = data.head(20)
		print(peek)

	@classmethod
	def dataRead(cls):
		data = read_csv(cls.filename,names=cls.names)
		print("Data shape")
		print(data.shape)

		types = data.dtypes
		print()
		print(f"DTypes: {types}")

		set_option('display.width',100)
		set_option('precision',3)
		description = data.describe()
		print(description)

		print("\nClassCounts")
		class_counts = data.groupby('class').size()
		print(class_counts)

		print("\nCorrelations")
		correlations = data.corr(method='pearson')
		print(correlations)

		print("\nSkew")
		skew = data.skew()
		print(skew)
		