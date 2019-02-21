from classes.project_iris import ProjectIrisTest
from classes.project_regression_boston import ProjectBostonTest
from classes.project_binaryclassification_mines_and_rocks import ProjectMinesAndRocks
#project selection
#FIRST PROJ, IRIS: 1
#REGRESSION MACHINE, Boston: 2
#CLASSIFICATION MACHINE, Mines and Rocks: 3
project = 3

#iris selected
if (project == 1): 
	#selects what to do in the project
	#Data Exploration: 1
	#Univariate plots: 2
	#Multivariate Plots: 3
	#EvaluateAlgorithms: 4

	project_option = 4
	#run project with option
	ProjectIrisTest.iris_project(project_option)

#Regression Machine
if (project == 2):
	
	#instantiating an instance of a ProjectBostonTest object
	bostonProj = ProjectBostonTest()
	#calling a method (function) of a ProjectBostonTest object
	bostonProj.runTest() 

	#selects what to do in the project
	#Data Exploration: 1
	#Univariate plots: 2
	#Multivariate Plots: 3
	#EvaluateAlgorithms: 4
	#EvaluateAlgorithmsEnsemble: 5

	project_option = 5
	#run method based on option option
	if (project_option == 1):
		bostonProj.dataExplore()
	elif (project_option == 2):
		bostonProj.uniPlots()
	elif (project_option == 3):
		bostonProj.multiplots()
	elif (project_option == 4):
		bostonProj.evaluateAlgorithm()
	elif (project_option == 5):
		bostonProj.evaluateAlgorithmEnsemble()

if (project == 3):

	#instantiating an instance of a ProjectBostonTest object
	mrProj = ProjectMinesAndRocks()
	#calling a method (function) of a ProjectBostonTest object
	mrProj.runTest() 

	#selects what to do in the project
	#Data Exploration: 1
	#Univariate plots: 2
	#Multivariate Plots: 3
	#EvaluateAlgorithms: 4
	#AlgorithmTuning: 5
	#AlgorithmsEnsemble: 6
	
	project_option = 6
	if (project_option ==1):
		mrProj.dataExplore()
	elif (project_option == 2):
		mrProj.uniPlots()
	elif (project_option == 3):
		mrProj.multiplots()
	elif (project_option == 4):
		mrProj.evaluateAlgorithm()
	elif (project_option == 5):
		mrProj.algorithmTuning()	
	elif (project_option == 6):
		mrProj.algorithmsEnsemble()









