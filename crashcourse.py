# String
data = 'hello world'
print(data[0])
print(len(data))
print(data)

#numbers
value = 123.1
print(value)
value = 10
print(value)

#boolean
a = True
b = False
print(a,b)

#Multiple Assignments
a,b,c = 1,2,3
print(a,b,c)

(1,2,3)
#No value
a = None
print(a)

#Flow Control
print("")
value = 99
if value >= 99:
	print('That is fast')
elif value > 200:
	print('That is too fast')
else:
	print('That that is safe')


#Loops
print("")
for i in range(10):
	print(i)

print("")
i = 0
while i < 10:
	print(i)
	i += 1

#tuples
a = (1,2,3)
print(a)

print("")
mylist = [1,2,3]
print("Zeroth Value: %d" % mylist[0])
mylist.append(4)

print("List Length: %d" % len(mylist))
for value in mylist:
	print(value)

###############
#dictionaries
mydict = {'a':1,'b':2,'c':3}
print("A value: %d" % mydict['a'])
mydict['a']=11
print("A value: %d" % mydict['a'])
print("Keys: %s" % mydict.keys())
print("Values: %s" %mydict.values())
for key in mydict.keys():
	print(mydict[key])


#functions
def mysum(x,y):
	return x+y

# Test sum function
print(mysum(1,3))

















