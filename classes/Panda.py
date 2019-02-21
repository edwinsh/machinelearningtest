import pandas as pd
import numpy as np


class PandaManager(object):

	def __init__(self):
		self.panda = None

	@classmethod 
	def usePD(cls):
		print("usePD Method")

		data = np.array(['a','b','c','d'])
		s = pd.Series(data)
		print(s)

	@classmethod 
	def useNP(cls):

		base_list = [0,1,2,3,4,5,6,7,8,9]
		print("\nbase_list ",base_list[0:9:2])
		exit()
		print("hello")

		a = np.array([1,2,3])
		print("\nOneDimensions ",a)

		b = np.array([[1,2,4],[4,5,6]])
		print("\nTwoDimensions ",b)

		c = np.array([1,2,3,4,5], ndmin = 2)
		print("\nTwoDimensions ",c)
		print("\nTwoDimensions ",c[0][0])

		d = np.array([1,2,3], dtype = complex)
		print("\bComplex ",d)

		print("\n\n")
		numpy_arrays = [a,b,c,d]
		for arr in numpy_arrays:
			print("\nNUMP Object Array: ",arr)
			print("NUMP Shape: ",arr.shape)
