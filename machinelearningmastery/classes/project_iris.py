
#load libraries
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#import warnings to remove warning complaints
import warnings
#warnings.filterwarnings("ignore",category=FutureWarning)

#Class use class methods
class ProjectIrisTest(object):

	filename = 'iris.csv'
	names = ['sepal-length','sepal-width','petal-length','petal-width','classtype']
	
	def __init__():
		self.name = 'test'
	
	@classmethod
	def iris_project(cls,option=1):
		
		#explore data
		if (option == 1):
			cls.dataExplore()
		#visualize univariate plots data
		elif (option == 2):
			cls.uniPlots()
		#visualize multivariateplots 
		elif (option == 3):
			cls.multiplots()
		#evaluate models
		elif (option == 4):
			cls.evaluateAlgorithms()

	@classmethod
	def dataExplore(cls):
		dataset = read_csv(cls.filename,names=cls.names)
		print("\n*****DATA EXPLORATION*****\n")
		print("Datashape:\n{}\n".format(dataset.shape))
		print("Dataset head(20):\n{}\n".format(dataset.head(20)))
		print("Dataset describe:\n{}\n".format(dataset.describe))
		print("Dataset distribution:\n{}\n".format(dataset.groupby('classtype').size()))

	@classmethod
	def uniPlots(cls):
		dataset = read_csv(cls.filename,names=cls.names)
		
		#plots in succession
		#boxplot
		dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
		pyplot.show()
		#histogram
		dataset.hist()
		pyplot.show()

	@classmethod
	def multiplots(cls):
		dataset = read_csv(cls.filename,names=cls.names)

		#plots in succession
		#scatter_matrix
		scatter_matrix(dataset)
		pyplot.show()

	@classmethod
	def evaluateAlgorithms(cls):
		dataset = read_csv(cls.filename,names=cls.names)
		array = dataset.values
		X = array[:,0:4]
		Y = array[:,4]
		validation_size = 0.20
		seed = 7
		X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size,random_state=seed)

		#Spot-Check Algorithms
		models = []
		#models.append(('LR',LogisticRegression()))
		models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
		models.append(('LDA',LinearDiscriminantAnalysis()))
		models.append(('KNN',KNeighborsClassifier()))
		models.append(('CART',DecisionTreeClassifier()))
		models.append(('NB',GaussianNB()))
		#models.append(('SVM',SVC()))
		models.append(('SVM',SVC(gamma='auto')))

		#evaluate each model in turn
		results = []
		names = []

		#iterates over models, splits training sets into 10 sets. Trains with 9 and validates with 1.
		for name, model in models:
			kfold = KFold(n_splits=10,random_state=seed)
			cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
			results.append(cv_results)
			names.append(name)
			msg = "Name:{}, Mean:{}, STD:({})".format(name, cv_results.mean(),cv_results.std())
			print(msg)

		#Change to true for boxplots
		shouldPlot = False
		if (shouldPlot == True): 
			#plot comparison
			fig = pyplot.figure()
			fig.suptitle('Algorithm Comparison')
			ax = fig.add_subplot(111)
			pyplot.boxplot(results)
			ax.set_xticklabels(names)
			pyplot.show()

		#predictions with validating against validation set
		knn = KNeighborsClassifier()
		knn.fit(X_train,Y_train)
		predictions = knn.predict(X_validation)
		print(accuracy_score(Y_validation, predictions))
		print(confusion_matrix(Y_validation,predictions))
		print(classification_report(Y_validation,predictions))
		print("\n")
		print("\nResults: {}".format(predictions))
		
		#view results
		print()
		counter = 0
		for pred, actual in zip(predictions,Y_validation):
			counter += 1
			if (pred == actual):
				correct = 'True'
			else:
				correct = '***False***'
			print(f"Prediction {counter}: PRED:{pred}, ACTUAL:{actual}, {correct}")



