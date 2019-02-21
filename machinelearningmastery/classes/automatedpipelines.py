
from pandas import read_csv

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

#linear classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#nonlinear classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

class autoPipes(object):

	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']


	def __init__(self):
		self.name = 'name'

	@classmethod
	def autoPipeline(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		#create pipeline
		estimators = []
		estimators.append(('standardize',StandardScaler()))
		estimators.append(('lda',LinearDiscriminantAnalysis()))
		model = Pipeline(estimators)
		#evaluate pipeline
		kfold = KFold(n_splits=10,random_state=7)
		results = cross_val_score(model,X,Y,cv=kfold)
		print(results.mean())

	@classmethod
	def featureExtractionPipe(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		#create feature union
		features = []
		features.append(('pca',PCA(n_components=3)))
		features.append(('select_best',SelectKBest(k=6)))
		feature_union = FeatureUnion(features)
		#create pipeline
		estimators = []
		estimators.append(('feature_union',feature_union))
		estimators.append(('lda',LogisticRegression()))
		model = Pipeline(estimators)
		#evaluate pipeline
		kfold = KFold(n_splits=10,random_state=7)
		results = cross_val_score(model,X,Y,cv=kfold)
		print(results.mean())		

