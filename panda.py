import numpy as np 
import pandas as pd 

d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
	'Age':pd.Series([24,26,25,23,30,29,33]),
	'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])
	}

df = pd.DataFrame(d)
print("Our data series is:")
print(df)
print()

#print("size is {}".format(df.iloc[0,:].size))

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])}

#Create a DataFrame
df = pd.DataFrame(d)
print(df)
print()
print("\nCOUNT is :\n{}\n\n\nSUM is :\n{}\n\n\nMean is :\n{}\n\nSTD is:\n{}\n\n".
	format(df.iloc[:,1:3].count(),df.iloc[:,1:3].sum(),df.iloc[:,1:3].mean(),df.iloc[:,1:3].std()) )

print("\n\nRunningTotals\n\n")
print("\nSUM is :\n{}\n\n\nPROD is :\n{}\n\n\nDescribe is :{}\n".
	format(df.iloc[:,1:3].cumsum(),df.iloc[:,1:3].prod(),df.iloc[:,:].describe(include='all')) )

"""
df[(df.Age >= 30) & (df.Age <= 30)jmj] = df[(df.Age >= 30) & (df.Age <= 30)]*2
print("Ourdata is {}".format(df.tail(3).iloc[0:2,:]))
print("Ourdata Types are {}".format(df.tail(3).iloc[0:2,:].dtypes))

a = 5
if (a > 2 and a < 7):
	print("A is between 2 and 7")
"""
print("\n\n\n")
def adder(ele1,ele2):
	return ele1+ele2

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
print("{}\n\n\n".format(df))
df = df.pipe(adder,2)
print(df)
print(df.apply(lambda x: x*2))
print(df.apply(lambda x: x*2, axis = 1))
print(df.apply(lambda x: x*3))
print(df.applymap(lambda x: x*3))
#print(df.apply(np.mean, axis = 1))


