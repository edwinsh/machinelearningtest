

from pandas import read_csv
from matplotlib import pyplot

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#linear classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#nonlinear classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class CompareAlgorithms(object):

	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

	def __init__(self):
		self.name = 'self'

	@classmethod
	def compareModels(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]

		#prepare models
		models = []
		models.append(('LR',LogisticRegression()))
		models.append(('LDA',LinearDiscriminantAnalysis()))
		models.append(('KNN',DecisionTreeClassifier()))
		models.append(('CART',LogisticRegression()))
		models.append(('NB',GaussianNB()))
		models.append(('SVM',SVC()))

		#evaluate each model in turn
		results = []
		names = []
		scoring = 'accuracy'
		for name, model in models:
			kfold = KFold(n_splits=10,random_state=7)
			cv_results = cross_val_score(model, X,Y,cv=kfold, scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "{} {} {}".format(name, cv_results.mean(),cv_results.std())
			print(msg)
		#boxplot algorithm comparison
		fig = pyplot.figure()
		fig.suptitle('Algorithm Comparison')
		ax = fig.add_subplot(111)
		pyplot.boxplot(results)
		ax.set_xticklabels(names)
		pyplot.show()





