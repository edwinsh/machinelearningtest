from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
import numpy

class UnderstanderVis(object):

	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']	

	#constructor
	def __init__(self):
		self.name = 'test'

	@classmethod
	def plotTestHist(cls):
		data = read_csv(cls.filename,names=cls.names)
		data.hist()
		pyplot.show()

	@classmethod
	def plotTestDensity(cls):
		data = read_csv(cls.filename,names=cls.names)
		data.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
		pyplot.show()

	@classmethod
	def plotTestBoxAndWhisker(cls):
		data = read_csv(cls.filename,names=cls.names)
		data.plot(kind='box',subplots=True,layout=(3,3),sharex=False)
		pyplot.show()

	@classmethod
	def plotTestCorrelation(cls):
		data = read_csv(cls.filename,names=cls.names)
		correlations = data.corr()
		#plot correlation matrix
		fig = pyplot.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(correlations,vmin=-1,vmax=1)
		fig.colorbar(cax)
		ticks = numpy.arange(0,9,1)
		ax.set_xticks(ticks)
		ax.set_yticks(ticks)
		ax.set_xticklabels(cls.names)
		ax.set_yticklabels(cls.names)
		pyplot.show()		

	@classmethod
	def plotTestScatterplot(cls):
		data = read_csv(cls.filename,names=cls.names)
		scatter_matrix(data)
		pyplot.show()