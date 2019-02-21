
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

class FeatureSelector(object):

	#class variables
	filename = 'pima-indians-diabetes.data.csv' 
	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

	#constructor
	def __init__(self):
		self.name = 'test'		

	@classmethod
	def univariateSelect(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		#feature extraction
		test = SelectKBest(score_func=chi2,k=4)
		fit = test.fit(X,Y)
		#summarize scores
		set_printoptions(precision=3)
		print(fit.scores_)
		features = fit.transform(X)

		#summarize selected features
		print(features[0:8,:])

		#summarize selected features
		print(features[0:5,:])

	@classmethod
	def recursiveFeature(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		# feature extraction
		model = LogisticRegression()
		rfe = RFE(model,3)
		fit = rfe.fit(X,Y)
		print("Num Features: {}".format(fit.n_features_))
		print("Selected Features: {}".format(fit.support_))
		print("Feature Ranking: {}".format(fit.ranking_))

	@classmethod
	def principalComponentAnalysis(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		#feature extraction
		pca = PCA(n_components=3)
		fit = pca.fit(X)
		#summarize components
		print("Explained Variance: {}".format(fit.explained_variance_ratio_))
		print(fit.components_)

	@classmethod
	def featureImportance(cls):
		dataframe = read_csv(cls.filename,names=cls.names)
		array = dataframe.values
		X = array[:,0:8]
		Y = array[:,8]
		#features extraction
		model = ExtraTreesClassifier()
		model.fit(X,Y)
		print(model.feature_importances_)

	







