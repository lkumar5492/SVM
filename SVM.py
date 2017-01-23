import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.io as sio
from svmutil import *
import time

def standardize(featureMatrix, normDict, featureNames):
	for x in featureNames:
		featureMatrix[x] = (featureMatrix[x] - normDict[x]["MEAN"])/(normDict[x]["SD"] * 1.0)
	return featureMatrix

def calculateNormDict(featureMatrix, featureNames):
	normDict = {}
	for x in featureNames:
		normDict.setdefault(x,{})
		normDict[x]["MEAN"] = featureMatrix[x].mean()
		normDict[x]["SD"] = featureMatrix[x].as_matrix().std()
	return normDict

def calculateTheta(featureMatrix, targetMatrix, regressor, lamda):
	if regressor == "LINEAR":
		innerProduct = np.dot(featureMatrix.transpose().as_matrix(),featureMatrix.as_matrix())
	elif regressor == "RIDGE":
		N = featureMatrix.columns.size
		identityMatrix = lamda * np.identity(N)
		innerProduct = np.dot(featureMatrix.transpose().as_matrix(),featureMatrix.as_matrix()) + identityMatrix

	inverseVal = np.linalg.pinv(innerProduct)
	theta = np.dot(np.dot(inverseVal,featureMatrix.transpose().as_matrix()),targetMatrix.as_matrix())
	return theta

def calculateMSE(featureMatrix, theta, targetMatrix):
	predictedTarget = np.dot(featureMatrix.as_matrix(), theta)
	squaredError = (predictedTarget.flatten() - targetMatrix["target"])**2
	sumOfSquaredError = squaredError.sum()
	meanSquareError = (sumOfSquaredError*1.0)/len(predictedTarget)
	return meanSquareError, sumOfSquaredError

def linearRegression(featureMatrix, targetMatrix, normDict, featureNames):

	#featureMatrix = standardize(featureMatrix, normDict, featureNames)
	theta = calculateTheta(featureMatrix, targetMatrix, "LINEAR", 0)
	MSE, sumOfSquaredError = calculateMSE(featureMatrix, theta, targetMatrix)
	return MSE, theta, sumOfSquaredError

def ridgeRegression(featureMatrix, targetMatrix, normDict, featureNames, lamda): #,trainingFeature, trainingTarget, testFeature, testTarget, normDict, featureNames):
	# #lamdaValues = [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
	# #print "==========   Ridge Regression ============"
	# for lamda in lamdaValues:
		# print "For LAMBDA = "+ str(lamda)
	theta = calculateTheta(featureMatrix, targetMatrix, "RIDGE", lamda)
	MSE = calculateMSE(featureMatrix, theta, targetMatrix)
	return MSE, theta

def sampledWithNDataSets(n, sizeOfDataSet):
	result = []
	SSEResult = []
	thetaList = []
	np.random.seed(123)
	#n = 100
	for i in range(0,n):
		thetaDict = {}
		MSEList = []
		SSEList = []
		data = np.random.uniform(-1,1.00000001, size =sizeOfDataSet)
		#print data
		noise = np.random.normal(0,np.sqrt(0.1), size=sizeOfDataSet)
		#print data
		target = copy.deepcopy(data)
		target = ((target**2) * 2) + noise
		#print target
		targetMatrix = pd.DataFrame(target, columns= ["target"])

		#For g1(x) = 1
		featureNames = []
		featureData = np.ones(sizeOfDataSet)
		featureMatrix = pd.DataFrame(featureData, columns= ["y"])
		mseG1, sumOfSquaredError = calculateMSE(featureMatrix, [1], targetMatrix)
		MSEList.append(mseG1)
		SSEList.append(sumOfSquaredError)
		thetaDict["g1"]=[1]
	

		#For g2(x) = w0
		featureNames = []
		featureData = np.ones(sizeOfDataSet)
		featureMatrix = pd.DataFrame(featureData, columns= ["x0"])
		normDict = calculateNormDict(featureMatrix, featureNames)
		fMatrix = featureMatrix.copy()
		mseG2, theta, sumOfSquaredError = linearRegression(fMatrix, targetMatrix, normDict, featureNames)
		MSEList.append(mseG2)
		SSEList.append(sumOfSquaredError)
		thetaDict["g2"]=theta.flatten()
		
		
		#For g3(x) = w0 + w1x
		featureNames = ["x"]
		featureMatrix = pd.DataFrame(data, columns= featureNames)
		featureMatrix["x0"] = 1
		normDict = calculateNormDict(featureMatrix, featureNames)
		fMatrix = featureMatrix.copy()
		# print fMatrix
		# print target
		mseG3, theta, sumOfSquaredError = linearRegression(fMatrix, targetMatrix, normDict, featureNames)
		MSEList.append(mseG3)
		SSEList.append(sumOfSquaredError)
		thetaDict["g3"]=theta.flatten()
		
		
		#For g4(x) = w0 + w1x + w2x^2
		featureMatrix["x2"] = data**2
		featureNames.append("x2")
		normDict = calculateNormDict(featureMatrix, featureNames)
		fMatrix = featureMatrix.copy()
		mseG4, theta, sumOfSquaredError = linearRegression(fMatrix, targetMatrix, normDict, featureNames)
		MSEList.append(mseG4)
		SSEList.append(sumOfSquaredError)
		thetaDict["g4"]=theta.flatten()
		

		#For g5(x) = w0 + w1x + w2x^2 + w3x^3
		featureMatrix["x3"] = data**3
		featureNames.append("x3")
		normDict = calculateNormDict(featureMatrix, featureNames)
		fMatrix = featureMatrix.copy()
		mseG5, theta, sumOfSquaredError = linearRegression(fMatrix, targetMatrix, normDict, featureNames)
		MSEList.append(mseG5)
		SSEList.append(sumOfSquaredError)
		thetaDict["g5"]=theta.flatten()
		

		#For g6(x) = w0 + w1x + w2x^2 + w3x^3 + w4x^4
		featureMatrix["x4"] = data**4
		featureNames.append("x4")
		normDict = calculateNormDict(featureMatrix, featureNames)
		fMatrix = featureMatrix.copy()
		mseG6, theta, sumOfSquaredError = linearRegression(fMatrix, targetMatrix, normDict, featureNames)
		MSEList.append(mseG6)
		SSEList.append(sumOfSquaredError)
		thetaDict["g6"]=theta.flatten()

		result.append(MSEList)
		SSEResult.append(SSEList)
		thetaList.append(thetaDict)

	result = np.array(result)
	SSEResult  = np.array(SSEResult)
	# print result[:,2].sum()
	print "==========   Histogram  ============"
	colNo = 0
	for i in range(1,7):
		print "Plotting histogram for g"+str(i)+" (Bins = 10): "
		subplot = plt.subplot(3,2,colNo+1)
		subplot.hist(result[:, i-1],10)
		subplot.set_xlabel("MSE for g"+str(i))
		subplot.set_ylabel("DataSets")
		colNo=colNo + 1
	plt.show()

	#print thetaList
	################# BIAS-SQUARE ####################
	data = np.random.uniform(-1,1.00000001, size =500)
	#print data

	bias = np.zeros(6)
	for x in data:
		epsilon = np.random.normal(0,np.sqrt(0.1))
		#For g1(x) = 1
		expectedHDx = 0.0
		for i in range(0,n):
			theta = thetaList[i]["g1"]
			#print theta
			xMatrix = [1]
			expectedHDx = expectedHDx +  np.dot(xMatrix, theta)
		expectedHDx = (expectedHDx * 1.0)/n
		expectedYx = 2*(x**2)
		pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
		probXY = pdf * (1.0/500)
		bias[0] = bias[0] + (((expectedHDx - expectedYx)**2) * probXY)
		
		#For g2(x) = w0
		expectedHDx = 0.0
		for i in range(0,n):
			theta = thetaList[i]["g2"]
			#print theta
			xMatrix = [1]
			expectedHDx = expectedHDx + np.dot(xMatrix, theta)
		expectedHDx = (expectedHDx * 1.0)/n
		expectedYx = 2*(x**2)
		pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
		probXY = pdf * (1.0/500)
		bias[1] = bias[1] + (((expectedHDx - expectedYx)**2) * probXY)

		#For g3(x) = w0 + w1x
		expectedHDx = 0.0
		for i in range(0,n):
			theta = thetaList[i]["g3"]
			# print theta
			xMatrix = [x,1]
			expectedHDx = expectedHDx + np.dot(xMatrix, theta)
		expectedHDx = (expectedHDx * 1.0)/n
		expectedYx = 2*(x**2)
		pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
		probXY = pdf * (1.0/500)
		bias[2] = bias[2] + (((expectedHDx - expectedYx)**2) * probXY)

		#For g4(x) = w0 + w1x + w2x^2
		expectedHDx = 0.0
		for i in range(0,n):
			theta = thetaList[i]["g4"]
			#print theta
			xMatrix = [x,1,x**2]
			expectedHDx = expectedHDx + np.dot(xMatrix, theta)
		expectedHDx = (expectedHDx * 1.0)/n
		expectedYx = 2*(x**2)
		pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
		probXY = pdf * (1.0/500)
		bias[3] = bias[3] + (((expectedHDx - expectedYx)**2) * probXY)

		#For g5(x) = w0 + w1x + w2x^2 + w3x^3
		expectedHDx = 0.0
		for i in range(0,n):
			theta = thetaList[i]["g5"]
			#print theta
			xMatrix = [x,1,x**2,x**3]
			expectedHDx = expectedHDx + np.dot(xMatrix, theta)
		expectedHDx = (expectedHDx * 1.0)/n
		expectedYx = 2*(x**2)
		pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
		probXY = pdf * (1.0/500)
		bias[4] = bias[4] + (((expectedHDx - expectedYx)**2) * probXY)

		#For g6(x) = w0 + w1x + w2x^2 + w3x^3 + w4x^4
		expectedHDx = 0.0
		for i in range(0,n):
			theta = thetaList[i]["g6"]
			#print theta
			xMatrix = [x,1,x**2,x**3,x**4]
			expectedHDx = expectedHDx + np.dot(xMatrix, theta)
		expectedHDx = (expectedHDx * 1.0)/n
		expectedYx = 2*(x**2)
		pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
		probXY = pdf * (1.0/500)
		bias[5] = bias[5] + (((expectedHDx - expectedYx)**2) * probXY)


	# print bias

	variance = np.zeros(6)
	for j in range(0,n):

		#bias = np.zeros(6)
		for x in data:
			epsilon = np.random.normal(0,np.sqrt(0.1))
			#For g1(x) = 1
			expectedHDx = 0.0
			for i in range(0,n):
				theta = thetaList[i]["g1"]
				#print theta
				xMatrix = [1]
				expectedHDx = expectedHDx + np.dot(xMatrix, theta)
			
			expectedHDx = (expectedHDx * 1.0)/n
			#expectedYx = 2*(x**2)
			pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
			probXY = pdf * (1.0/500)
			#bias[0] = bias[0] + (((expectedHDx - expectedYx)**2) * probXY)
			hDx = np.dot(xMatrix, thetaList[j]["g1"])
			variance[0] = variance[0] + (((hDx - expectedHDx)**2) * probXY)
			
			#For g2(x) = w0
			expectedHDx = 0.0
			for i in range(0,n):
				theta = thetaList[i]["g2"]
				#print theta
				xMatrix = [1]
				expectedHDx = expectedHDx + np.dot(xMatrix, theta)
			expectedHDx = (expectedHDx * 1.0)/n
			expectedYx = 2*(x**2)
			pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
			probXY = pdf * (1.0/500)
			#bias[1] = bias[1] + (((expectedHDx - expectedYx)**2) * probXY)
			hDx = np.dot(xMatrix, thetaList[j]["g2"])
			variance[1] = variance[1] + (((hDx - expectedHDx)**2) * probXY)

			#For g3(x) = w0 + w1x
			expectedHDx = 0.0
			for i in range(0,n):
				theta = thetaList[i]["g3"]
				# print theta
				xMatrix = [x,1]
				expectedHDx = expectedHDx + np.dot(xMatrix, theta)
			expectedHDx = (expectedHDx * 1.0)/n
			expectedYx = 2*(x**2)
			pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
			probXY = pdf * (1.0/500)
			#bias[2] = bias[2] + (((expectedHDx - expectedYx)**2) * probXY)
			hDx = np.dot(xMatrix, thetaList[j]["g3"])
			variance[2] = variance[2] + (((hDx - expectedHDx)**2) * probXY)

			#For g4(x) = w0 + w1x + w2x^2
			expectedHDx = 0.0
			for i in range(0,n):
				theta = thetaList[i]["g4"]
				#print theta
				xMatrix = [x,1,x**2]
				expectedHDx = expectedHDx + np.dot(xMatrix, theta)
			expectedHDx = (expectedHDx * 1.0)/n
			expectedYx = 2*(x**2)
			pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
			probXY = pdf * (1.0/500)
			#bias[3] = bias[3] + (((expectedHDx - expectedYx)**2) * probXY)
			hDx = np.dot(xMatrix, thetaList[j]["g4"])
			variance[3] = variance[3] + (((hDx - expectedHDx)**2) * probXY)

			#For g5(x) = w0 + w1x + w2x^2 + w3x^3
			expectedHDx = 0.0
			for i in range(0,n):
				theta = thetaList[i]["g5"]
				#print theta
				xMatrix = [x,1,x**2,x**3]
				expectedHDx = expectedHDx + np.dot(xMatrix, theta)
			expectedHDx = (expectedHDx * 1.0)/n
			expectedYx = 2*(x**2)
			pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
			probXY = pdf * (1.0/500)
			#bias[4] = bias[4] + (((expectedHDx - expectedYx)**2) * probXY)
			hDx = np.dot(xMatrix, thetaList[j]["g5"])
			variance[4] = variance[4] + (((hDx - expectedHDx)**2) * probXY)

			#For g6(x) = w0 + w1x + w2x^2 + w3x^3 + w4x^4
			expectedHDx = 0.0
			for i in range(0,n):
				theta = thetaList[i]["g6"]
				#print theta
				xMatrix = [x,1,x**2,x**3,x**4]
				expectedHDx = expectedHDx + np.dot(xMatrix, theta)
			expectedHDx = (expectedHDx * 1.0)/n
			expectedYx = 2*(x**2)
			pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
			probXY = pdf * (1.0/500)
			#bias[5] = bias[5] + (((expectedHDx - expectedYx)**2) * probXY)
			hDx = np.dot(xMatrix, thetaList[j]["g6"])
			variance[5] = variance[5] + (((hDx - expectedHDx)**2) * probXY)

			
	for i in range(0,6):
		variance[i] = variance[i] * (1.0/n)


	for i,b in enumerate(bias):
		print "Sum of squared error for g" + str(i+1) + ": " + str(SSEResult[:,int(i)].sum())
		print "Bias for g" + str(i+1) + ": " + str(b)
		print "Variance for g" + str(i+1) + ": " + str(variance[i])

	# print variance

def sampledWithNDataSetsUsingRidge(n, sizeOfDataSet):

	result = []
	thetaList = []
	np.random.seed(123)
	#n = 100
	lamdaValues = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
	for i in range(0,n):
		thetaDict = {}
		MSEList = []
		data = np.random.uniform(-1,1.00000001, size =sizeOfDataSet)
		#print data
		noise = np.random.normal(0,np.sqrt(0.1), size=sizeOfDataSet)
		#print data
		target = copy.deepcopy(data)
		target = ((target**2) * 2) + noise
		#print target
		targetMatrix = pd.DataFrame(target, columns= ["target"])

		#For h(x) = w0 + w1x + w2x^2
		featureNames = ["x"]
		featureMatrix = pd.DataFrame(data, columns= featureNames)
		featureMatrix["x0"] = 1
		featureMatrix["x2"] = data**2
		featureNames.append("x2")
		normDict = calculateNormDict(featureMatrix, featureNames)
		fMatrix = featureMatrix.copy()

		for lamda in lamdaValues:
			mse, theta = ridgeRegression(featureMatrix, targetMatrix, normDict, featureNames, lamda)
			MSEList.append(mse)
			thetaDict[lamda]=theta.flatten()
				
		result.append(MSEList)
		thetaList.append(thetaDict)

	result = np.array(result)
	for i in range(0,7):
		print "Sum-square-error for lambda = "+ str(lamdaValues[i]) +" : "+ str(result[:,i].sum())
		
	################# BIAS-SQUARE ####################
	data = np.random.uniform(-1,1.00000001, size =500)
	#print data

	bias = np.zeros(7)
	for x in data:
		epsilon = np.random.normal(0,np.sqrt(0.1))
		
		#For h(x) = w0 + w1x + w2x^2
		for index,lamda in enumerate(lamdaValues):
			expectedHDx = 0.0
			for i in range(0,n):
				theta = thetaList[i][lamda]
				#print theta
				xMatrix = [x,1,x**2]
				expectedHDx = expectedHDx + np.dot(xMatrix, theta)
			expectedHDx = (expectedHDx * 1.0)/n
			expectedYx = 2*(x**2)
			pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
			probXY = pdf * (1.0/500)
			bias[index] = bias[index] + (((expectedHDx - expectedYx)**2) * probXY)

	#print bias

	variance = np.zeros(7)
	for j in range(0,n):
		for x in data:
			epsilon = np.random.normal(0,np.sqrt(0.1))
			#For h(x) = w0 + w1x + w2x^2
			for index,lamda in enumerate(lamdaValues):
				expectedHDx = 0.0
				for i in range(0,n):
					theta = thetaList[i][lamda]
					#print theta
					xMatrix = [x,1,x**2]
					expectedHDx = expectedHDx + np.dot(xMatrix, theta)
				expectedHDx = (expectedHDx * 1.0)/n
				expectedYx = 2*(x**2)
				pdf = norm.pdf(expectedYx + epsilon, loc = expectedYx, scale = np.sqrt(0.1))
				probXY = pdf * (1.0/500)
				hDx = np.dot(xMatrix, thetaList[j][lamda])
				variance[index] = variance[index] + (((hDx - expectedHDx)**2) * probXY)
			
	for i in range(0,7):
		variance[i] = variance[i] * (1.0/n)

	#print variance
	for i,b in enumerate(bias):
		print "Bias for lambda = " + str(lamdaValues[i]) + ": " + str(b)
		print "Variance for lambda = " + str(lamdaValues[i]) + ": " + str(variance[i])

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

def svm():
	###########  SVM ###################
	trainData = sio.loadmat("phishing-train")

	trainFeatures = trainData["features"]
	trainTarget = trainData["label"]

	trainPPFeatures = svmPreprocessing(trainFeatures)
	prob = svm_problem(trainTarget.flatten(), trainPPFeatures)

	testData = sio.loadmat("phishing-test")
	testFeatures = testData["features"]
	testTarget = testData["label"]
	testPPFeatures = svmPreprocessing(testFeatures)

	print "================= LINEAR SVM ==================="
	cValues = []
	for i in range(-6,3):
		cValues.append(4**i)
	# print "For Training Data:"
	# print ""
	maxAccuracy = 0.0
	maxCVal = 0
	start = time.time()
	for c in cValues:
		param = svm_parameter("-c "+ str(c)+" -v 3 -t 0 -q")
		print "3-fold Cross-Validation Accuracy for c=" + str(c) +": " 
		accuracy = svm_train(prob, param)
		if(accuracy > maxAccuracy):
			maxAccuracy = accuracy
			maxCVal = c

	print "Best c:" + str(maxCVal)

	param = svm_parameter("-c "+ str(maxCVal)+" -t 0 -q")
	model = svm_train(prob, param)	

	svm_predict(testTarget.flatten(), testPPFeatures, model)
	end = time.time()
	print "Average Training Time : " + str(end - start)
	
	print "================= KERNEL SVM ==================="
	cValues = []
	for i in range(-3,8):
		cValues.append(4**i)
	degree = [1,2,3]
	# print "POLYNOMIAL KERNEL ::::::::::"
	# print "For Training Data:"
	# print ""
	maxAccuracy = 0.0
	maxCVal = 0
	maxDVal = 0
	maxGVal = 0

	start = time.time()
	for c in cValues:
		for d in degree:
			param = svm_parameter("-c "+ str(c)+" -v 3 -t 1 -d "+str(d)+ " -q")
			# print "3-fold Cross-Validation Accuracy for c=" + str(c) +" and degree= "+ str(d) + " : " 
			accuracy = svm_train(prob, param)
			if accuracy > maxAccuracy:
				maxAccuracy = accuracy
				maxCVal = c
				maxDVal = d
				maxGVal = 0

	end = time.time()
	print "Average Training Time : " + str(end - start)
	gValues = []
	for i in range(-7,0):
		gValues.append(4**i)

	# print ""
	# print "RBF KERNEL ::::::::::"
	# print "For Training Data:"
	# print ""
	start = time.time()
	for c in cValues:
		for g in gValues:
			param = svm_parameter("-c "+ str(c)+" -v 3 -t 2 -g "+str(g)+ " -q")
			print "3-fold Cross-Validation Accuracy for c=" + str(c) +" and gamma= "+ str(g) + " : " 
			accuracy = svm_train(prob, param)
			if accuracy > maxAccuracy:
				maxAccuracy = accuracy
				maxCVal = c
				maxGVal = g
				maxDVal = 0
		
	#print cValues
	print ""
	print "Best c:" + str(maxCVal)
	print "Max Train Accuracy:" + str(maxAccuracy)
	if maxDVal == 0:
		print "Best g:" + str(maxGVal)
		param = svm_parameter("-c "+ str(maxCVal)+" -t 2 -g "+str(maxGVal)+ " -q")
	elif maxGVal == 0:
		print "Best d:" + str(maxDVal)
		param = svm_parameter("-c "+ str(maxCVal)+" -t 1 -d "+str(maxDVal)+ " -q")

	model = svm_train(prob, param)	

	svm_predict(testTarget.flatten(), testPPFeatures, model)
	end = time.time()
	print "Average Training Time : " + str(end - start)
	

if __name__ == "__main__":

	print "100 datasets with 10 samples each:"
	sampledWithNDataSets(100, 10)

	print "100 datasets with 100 samples each:"
	sampledWithNDataSets(100, 100)

	sampledWithNDataSetsUsingRidge(100, 100)

	svm()

	
