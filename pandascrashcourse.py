# series

import numpy
import pandas
myarray = numpy.array([1,2,3])
rownames = ['a','b','c']
myseries = pandas.Series(myarray, index=rownames)
print(myseries)

myarray = numpy.array([[1,2,3],[4,5,6]])
rownames = ['a','b']
colnames = ['one','two','three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)

print("")
print("one row: \n%s" % mydataframe.loc['a',:])
print("")
print("one column: \n%s" % mydataframe['one'])
print("one column: \n%s" % mydataframe.one)
print("")
print("one field: %d" % mydataframe.loc['a','three'])