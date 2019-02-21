
import csv
import numpy
from numpy import loadtxt
from urllib.request import urlopen
from pandas import read_csv

class FileLoader(object):
	def __init__(self):
		self.name = 'test'

	#decorator
	@classmethod
	def test(cls):
		print("FileloaderTest")

	@classmethod
	def loadCSV(cls):
		filename = 'pima-indians-diabetes.data.csv'
		#raw_data = open(filename, 'rb')
		raw_data = open(filename, 'r')
		reader = csv.reader(raw_data,delimiter=',',quoting=csv.QUOTE_NONE)
		
		#readline from file
		#line_count = 0
		#for row in reader:
		#	print(f"Row {line_count} is {row}")
		#	line_count+=1

		x = list(reader)
		data = numpy.array(x).astype('float')
		print(data.shape)


		#load csv with numpy
		filename = 'pima-indians-diabetes.data.csv'
		raw_data = open(filename, 'r')
		data = loadtxt(raw_data, delimiter=",")
		print(data.shape)

		#load csv with url using numpy
		url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
		raw_data = urlopen(url)
		dataset = loadtxt(raw_data, delimiter=",")
		print(dataset.shape)

		#load csv using pandas
		#from pandas import read_csv
		filename = 'pima-indians-diabetes.data.csv'
		names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
		data = read_csv(filename,names=names)
		print(data.shape)

		#load csv with url using numpy
		url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
		names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
		data = read_csv(url,names=names)
		print(data.shape)






