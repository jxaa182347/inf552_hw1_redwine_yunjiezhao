import sys
import random
from sklearn.neural_network import MLPClassifier

def getDist(data):
	quaSta = {}
	rowSize = len(data)
	colSize = len(data[0])
	for i in range(rowSize):
		if data[i][colSize-1] in quaSta:
			quaSta[data[i][colSize-1]] += 1
		else:
			quaSta[data[i][colSize-1]] = 1
	return quaSta

def getData(filedir, k_fold):
	dataLabel = []
	data = []
	dataCleaned = []
	dataTraining = []
	dataTest = []

	fin = open(filedir,"rb")
	readfile = fin.readlines()
	fin.close()	

	lineno = 0
	for line in readfile:
		length = len(line)
		if lineno == 0:
			crux = 0
			precrux = -1
			while(crux != length):
				if line[crux] == ';':
					dataLabel.append(line[precrux+2:crux-1])			
					precrux = crux				
				crux += 1

		else:
			tempList = []		
			crux = 0
			precrux = -1
			while(crux != length):
				if line[crux] == ';' or crux == length - 1:
					tempList.append(float(line[precrux+1:crux]))			
					precrux = crux
				crux += 1
			data.append(tempList)
		lineno += 1

	for i in range(len(data)):
		if data[i] in dataCleaned:
			continue
		else:
			dataCleaned.append(data[i])

	#dataCleanedDist = getDist(dataCleaned)
	#print dataCleanedDist

	lineSum = len(dataCleaned)
	testSize = lineSum / k_fold
	trainingSize = lineSum - testSize
	tempRan = random.sample(range(0,lineSum),trainingSize)

	for i in range(len(dataCleaned)):
		if i in tempRan:
			dataTraining.append(dataCleaned[i])
		else:
			dataTest.append(dataCleaned[i])
	
	return dataTraining, dataTest

def sliceData(data):
	rowSize = len(data)
	colSize = len(data[0])
	dataIn = data
	dataOut = []

	for i in range(rowSize):
		dataOut.append(dataIn[i][colSize-1])
		del dataIn[i][colSize-1]
	
	return dataIn, dataOut

def getScore(randomTimes, k_fold):
	scoreSum = 0.0
	for randomCount in range(randomTimes):
		dataTraining, dataTest = getData("redwine-data",k_fold)
		dataIn, dataOut = sliceData(dataTraining)

		clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
		clf.fit(dataIn, dataOut) 
		MLPClassifier(activation='relu', algorithm='l-bfgs', alpha=1e-05,
		       batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
		       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
		       learning_rate_init=0.001, max_iter=200, momentum=0.9,
		       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
		       tol=0.0001, validation_fraction=0.1, verbose=False,
		       warm_start=False)

		testIn, testOut = sliceData(dataTest)
		modelOut = clf.predict(testIn)
		testSize = len(testOut)

		"""
		testDist = {}
		modelDist = {}
		for i in range(testSize):
			if testOut[i] in testDist:
				testDist[testOut[i]] += 1
			else:
				testDist[testOut[i]] = 1
		for i in range(testSize):
			if modelOut[i] in modelDist:
				modelDist[modelOut[i]] += 1
			else:
				modelDist[modelOut[i]] = 1
		print testDist
		print modelDist
		"""

		count = 0
		for i in range(testSize):
			if testOut[i] == modelOut[i]:
				count += 1

		score = float(count) / float(testSize)
		#print score
		scoreSum += score
	avgScore = scoreSum / float(randomTimes)
	return avgScore

def getResult(kfold, times):
	maxKfold = kfold
	randomTimes = times
	result = []
	i_k = 2
	while(i_k <= maxKfold):
		avgScore = getScore(randomTimes,i_k)
		result.append(avgScore)
		i_k += 1
	print "---- MLPClassifier Evaluation ----\n K-fold Cross Validation\n < Random Times:"+str(randomTimes)+", K:From 2 to "+str(maxKfold)+" >\n Each Accuracy as below:"
	print result

getResult(10, 100)
