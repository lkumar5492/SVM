import sys
import scipy.io as sio
from svmutil import *
import pandas as pd
import numpy as np

def	svmPreprocessing(features):
	#print trainFeatures
	featuresDF = pd.DataFrame(features, columns = None)
	#print trainFeaturesDF
	indexWith3Values = [1, 6, 7, 13, 14, 25, 28]
	for i in indexWith3Values:
		indexList = featuresDF.loc[featuresDF[i] == -1][i].index
		col = str(i) + "_m1"
		colList = np.zeros(featuresDF.index.size)
		for index in indexList:
			colList[index] = 1
		featuresDF[col] = colList

		indexList = featuresDF.loc[featuresDF[i] == 0][i].index
		col = str(i) + "_0"
		colList = np.zeros(featuresDF.index.size)
		for index in indexList:
			colList[index] = 1
		featuresDF[col] = colList

		indexList = featuresDF.loc[featuresDF[i] == 1][i].index
		col = str(i) + "_p1"
		colList = np.zeros(featuresDF.index.size)
		for index in indexList:
			colList[index] = 1
		featuresDF[col] = colList

		featuresDF.drop(i,axis=1, inplace=True)
	#print trainFeaturesDF
	featuresDF.replace(-1, 0, inplace=True)
	#print trainFeaturesDF

	featuresDF.columns = range(0,44)
	#print featuresDF
	return featuresDF.T.to_dict().values()

if __name__ == "__main__":

	trainData = sio.loadmat("phishing-train")
	trainFeatures = trainData["features"]
	trainTarget = trainData["label"]
	trainPPFeatures = svmPreprocessing(trainFeatures)
	prob = svm_problem(trainTarget.flatten(), trainPPFeatures)

	testData = sio.loadmat("phishing-test")
	testFeatures = testData["features"]
	testTarget = testData["label"]
	testPPFeatures = svmPreprocessing(testFeatures)

	maxCVal = 16384
	maxGVal = 0.25
	param = svm_parameter("-c "+ str(maxCVal)+" -t 2 -g "+str(maxGVal)+ " -q")
	model = svm_train(prob, param)	

	print "BEST KERNEL: RBF KERNEL with c = "+ str(maxCVal) + " g = "+ str(maxGVal)
	print "Test Set Accuracy:" 
	svm_predict(testTarget.flatten(), testPPFeatures, model)
