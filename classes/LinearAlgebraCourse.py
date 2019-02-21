
from numpy import array
from numpy.linalg import inv

class LinearAlgebra(object):
	def __init__(self):
		self.name = 'linAlg'
	
	@classmethod 
	def vectorTest(cls):
		v = array([1,2,3])
		print(v)

	@classmethod 
	def vectorMath(cls):
		a = array([1,2,3])
		print(f"ArrayA: {a}")
		b = array([1,2,3])
		print(f"ArrayB: {b}")
		c = a*b
		print(f"ArrayC: c=a*b, {c}")
		d = a+b
		print(f"ArrayD: d=a+b, {d}")
		e = a-b
		print(f"ArrayE: e=a-b, {e}")
		f = a/b
		print(f"ArrayF: f=a/b, {f}")

	@classmethod 
	def matrixMath(cls):
		A = array([[1,2,3],[4,5,6]])
		print(f"Matrix A: {A}")

		B = array([[1,2,3],[4,5,6]])
		print(f"Matrix B: {B}")

		C = A + B 
		print(f"Matrix C: C=A+B {C}")

		A = array([[1,2],[3,4],[5,6]])
		print(A)
		B = array([[1,2],[3,4]])
		print(B)
		C = A.dot(B)
		print(f"Matrix C: C=A.dot(B)\n{C}")

	@classmethod
	def matrixOperations(cls):
		A = array([[1.0,2.0],[3.0,4.0]])
		print(A)
		B = inv(A)
		print(B)