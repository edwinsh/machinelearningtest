from classes.pythoncrash import PythonCrash
from classes.fileloading import FileLoader
from classes.understand import Understander
from classes.understandVis import UnderstanderVis
from classes.prepdataML import MLDataPrepper
from classes.featureselect import FeatureSelector
from classes.evaluateAlgorithm import algEvaluator
from classes.performance import PerformanceMetrics
from classes.spotcheckclass import SpotCheckClassification
from classes.spotcheckregress import SpotCheckRegression
from classes.comparealgorithms import CompareAlgorithms
from classes.automatedpipelines import autoPipes
from classes.improveperformance import ensemblePerformance
from classes.algorithmtuning import AlgorithmTuner
from classes.saveandload import SaveAndLoad

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

#Python Crash Course
#PythonCrash.testPython()
#PythonCrash.flowTest()
#PythonCrash.numpyTest()
#PythonCrash.matplotlibTest()
#PythonCrash.pandaTest()

#CSV File load 
#FileLoader.test()
#FileLoader.loadCSV()

#Dataset Exploration
#Understander.utest()
#Understander.dataPeek()
#Understander.dataRead()

#Dataset Visualization
#UnderstanderVis.plotTestHist()
#UnderstanderVis.plotTestDensity()
#UnderstanderVis.plotTestBoxAndWhisker()
#UnderstanderVis.plotTestCorrelation()
#UnderstanderVis.plotTestScatterplot()

#Data Scaling
#MLDataPrepper.rescaleData()

#Feature Selection
#FeatureSelector.univariateSelect()
#FeatureSelector.recursiveFeature()
#FeatureSelector.principalComponentAnalysis()
#FeatureSelector.featureImportance()

#Algorithm Evaluation
#algEvaluator.trainTestSplit()
#algEvaluator.kfoldCross()
#algEvaluator.leaveOneOutCross()
#algEvaluator.repeatedRandomTrainSplits()

#Performance Metrics
#PerformanceMetrics.classificationAccuracy()
#PerformanceMetrics.logarithmicLoss()
#PerformanceMetrics.areaUnderCurve()
#PerformanceMetrics.confusionMatrix()
#PerformanceMetrics.classificationReport()
#PerformanceMetrics.meanAbsoluteError()
#PerformanceMetrics.meanSquaredError()
#PerformanceMetrics.rSquared()

#spotcheck classification
#SpotCheckClassification.logisticRegression()
#SpotCheckClassification.linearDiscriminantAnalysis()
#SpotCheckClassification.kNearestNeighbors()
#SpotCheckClassification.naiveBayesClass()
#SpotCheckClassification.decisionTree()
#SpotCheckClassification.supportVectorMachine()

#spotcheck regression
#SpotCheckRegression.linearRegression()
#SpotCheckRegression.ridgeRegression()
#SpotCheckRegression.lassoRegression()
#SpotCheckRegression.elasticNetRegression()
#SpotCheckRegression.kneighborsRegression()
#SpotCheckRegression.decisionTreeRegression()
#SpotCheckRegression.svrRegression()

#compare algorithms
#CompareAlgorithms.compareModels()

#auto pipelines
#autoPipes.autoPipeline()
#autoPipes.featureExtractionPipe()

#ensemble performance
#ensemblePerformance.baggedTrees()
#ensemblePerformance.randomForest()
#ensemblePerformance.extraTrees()
#ensemblePerformance.adaBoost()
#ensemblePerformance.gradBoost()
#ensemblePerformance.votingEnsemble()

#AlgorithmTuner
#AlgorithmTuner.gridSearch()
#AlgorithmTuner.randomSearch()

#Save and Load
#SaveAndLoad.pickleTest()
SaveAndLoad.joblibTest()




