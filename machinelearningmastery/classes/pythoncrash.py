
import numpy
import matplotlib.pyplot as plt 
import pandas

class PythonCrash(object):
	def __init(self):
		self.name = 'myname'

	@classmethod
	def testPython(cls):
		print("helloworld test")

		data = 'hello world'
		print(data[0])
		print(len(data))
		print(data)

		a = True
		b = False
		print(a,b)
		a,b,c = 1,2,3
		print(a,b,c)

		a = None
		print(a)


	@classmethod
	def flowTest(cls):
		value = 99
		if (value == 99):
			print('That is fast')
		elif (value > 200):
			print('That is too fast')
		else:
			print('That is safe')

		#for loop
		for i in range(10):
			print(i)

		for i in range(5,10):
			print(i)

		for i in range(5,10,2):
			print(i)

		print("\n\nWHILE\n\n")

		i = 0
		while i < 10 :
			print(i)
			i += 1

		print ("\n\n")
		a = (1,2,3)
		print(a)

		mylist = [1,2,3]
		print("Zeroth Value: {}".format(mylist[0]))
		mylist.append(4)
		print("List Length: {}".format(len(mylist)))
		for value in mylist:
			print(value)

		print("\n\nDictionary\n\n")
		mydict = {'a':1,'b':2,'c':3}
		print("A value: {}".format(mydict['a']))
		mydict['a']=11
		print("A value: {}".format(mydict['a']))
		print("Keys: {}".format(mydict.keys()))
		print("values: {}".format(mydict.values()))
		for key in mydict.keys():
			print("Key: {}, Value: {}".format(key, mydict[key]))

		sumvalue = cls.mysum(5,6)
		print("5+6={}".format(sumvalue))

	@classmethod
	def mysum(cls,x,y):
		return x+y

	@classmethod
	def numpyTest(cls):
		mylist = [1,2,3]
		myarray = numpy.array(mylist)
		print(myarray)
		print(myarray.shape)

		#access data
		mylist = [[1,2,3],[4,5,6]]
		myarray = numpy.array(mylist)
		print(myarray)
		print(myarray.shape)
		print("First row: {}".format(myarray[0]))
		print("Last row: {}".format(myarray[1]))
		print("Specific row and col: {}".format(myarray[0,2]))
		print("Whole col: {}".format(myarray[:,2]))

		#arithmetic
		myarray1 = numpy.array([2,2,2])
		myarray2 = numpy.array([3,3,3])
		print("Addition: {}".format(myarray1 + myarray2))
		print("Multiplication: {}".format(myarray1*myarray2))

	@classmethod
	def matplotlibTest(cls):
		myarray = numpy.array([1,2,3])
		plt.plot(myarray)
		plt.xlabel('some x axis')
		plt.ylabel('some y axis')
		plt.show()

		x = numpy.array([1,2,3])
		y = numpy.array([2,4,6])
		plt.scatter(x,y)
		plt.xlabel('some x axis')
		plt.ylabel('some y axis')
		plt.show()

	@classmethod 
	def pandaTest(cls):
		myarray = numpy.array([1,2,3])
		rownames = ['a','b','c']
		myseries = pandas.Series(myarray, index=rownames)
		print(myseries)
		
		print("Myseries[0] {}".format(myseries[0]))
		print("Myseries['a'] {}".format(myseries['a']))
		print("Myseries\n{}".format(myseries))

		print("\n\nDataFrame\n\n")
		myarray = numpy.array([[1,2,3],[4,5,6]])
		rownames = ['a','b']
		colnames = ['one','two','three']
		mydataframe = pandas.DataFrame(myarray,index=rownames,columns=colnames)
		print(mydataframe)

		print("Method 1:")
		print("one column:\n{}".format(mydataframe['one']))
		print("Method 2:")
		print("one columns:\n{}".format(mydataframe.one))


		print("\n\nRow\nMethod 1:")
		print("one row:\n{}".format(mydataframe.loc['a']))
		print("Method 2:")
		print("one row:\n{}".format(mydataframe.loc['a',]))
		print("Method 3:")
		print("one row:\n{}".format(mydataframe.loc['a',:]))
		print("Method 3:")
		print("one row:\n{}".format(mydataframe.iloc[0,:]))










