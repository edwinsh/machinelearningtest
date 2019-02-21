#basic line plot
import matplotlib.pyplot as plt 
import numpy
myarray = numpy.array([1,2,3])
plt.plot(myarray)
plt.xlabel('some x axis')
plt.ylabel('some y axis')
plt.show()

import matplotlib.pyplot as plt 
import numpy
x = numpy.array([1,2,3])
y = numpy.array([4,5,6])
plt.scatter(x,y)
plt.xlabel('some x axis')
plt.ylabel('some y axis')
plt.show()

