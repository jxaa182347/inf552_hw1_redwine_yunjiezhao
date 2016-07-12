import numpy as np
import csv
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier

def readata(filedir, SIZE, YEAR, MOUTH, DAY, EVENT, CATE, ZIP):
	dict_time = {}
	dataTrain = []
	dataResult = []
	dataHead = []
	with open (filedir) as f:
		f_csv = csv.reader(f)
		headers = next(f_csv)
		dataHead.append(headers[5]) #CRIME_YEAR
		dataHead.append(headers[3]) #MOUTH
		dataHead.append(headers[4]) #DAY
		dataHead.append(headers[12]) #EVENT
		dataHead.append(headers[6]) #CRIME_CATEGORY_NO
		dataHead.append(headers[9]) #ZIP
		dataHead.append(headers[10]) #LATITUDE
		dataHead.append(headers[11]) #LONGITUDE
		dataHead.append(headers[2]) #TIME_PERIOD
		#print dataHead
		k = 0
		datasize = 0
		for row in f_csv:
			if SIZE != 'all':
				if k == SIZE:
					break
				k += 1
			dataList = []
			if float(row[5]) == YEAR or YEAR == 'all':
				if float(row[3]) == MOUTH or MOUTH == 'all':
					if float(row[4]) == DAY or DAY == 'all':
						if float(row[12]) == EVENT or EVENT == 'all':
							if float(row[6]) == CATE or CATE == 'all':
								if float(row[9]) == ZIP or ZIP == 'all':
									dataList.append(float(row[5])) #CRIME_YEAR
									dataList.append(float(row[3])) #MOUTH
									dataList.append(float(row[4])) #DAY
									dataList.append(float(row[12])) #EVENT
									dataList.append(float(row[6])) #CRIME_CATEGORY_NO
									dataList.append(float(row[9])) #ZIP
									dataList.append(float(row[10])) #LATITUDE
									dataList.append(float(row[11])) #LONGITUDE
									dataTrain.append(dataList)
									"""
									if int(row[2]) <= 3 and int(row[2]) >= 1:
										temp = 1.0
										dataResult.append(temp)  #TIME_PERIOD
										if float(temp) in dict_time:
											dict_time[temp] += 1
										else:
											dict_time[temp] = 1
									if int(row[2]) <= 6 and int(row[2]) >= 4:
										temp = 2.0
										dataResult.append(temp)  #TIME_PERIOD
										if float(temp) in dict_time:
											dict_time[temp] += 1
										else:
											dict_time[temp] = 1
									if int(row[2]) <= 9 and int(row[2]) >= 7:
										temp = 3.0
										dataResult.append(temp)  #TIME_PERIOD
										if float(temp) in dict_time:
											dict_time[temp] += 1
										else:
											dict_time[temp] = 1
									if int(row[2]) <= 12 and int(row[2]) >= 10:
										temp = 4.0
										dataResult.append(temp)  #TIME_PERIOD
										if float(temp) in dict_time:
											dict_time[temp] += 1
										else:
											dict_time[temp] = 1
									"""
									dataResult.append(float(row[2])) #TIME_PERIOD
									if float(row[2]) in dict_time:
										dict_time[float(row[2])] += 1
									else:
										dict_time[float(row[2])] = 1
									
									datasize += 1
	print 'Time Distribution:', dict_time
	print 'DataSize:', datasize
	return dataTrain, dataResult

def MLRClassifier(x,y):
	clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
				C=1.0, fit_intercept=True, intercept_scaling=1, 
				class_weight=None, random_state=None, solver='lbfgs', 
				max_iter=5000, multi_class='multinomial', verbose=0, warm_start=False, n_jobs=1)

	clf.fit(x,y)
	
	#ypre = clf.predict(x)
	#print x
	#print ypre
	#score = accuracy_score(ypre,y)
	#print score

	scores = cross_validation.cross_val_score(clf, x, y, cv=5)
	result = scores.mean()

	#test = np.array(x[0]).reshape(1,8)
	#print test
	#pro = clf.predict_proba(test)
	#print pro
	#target = clf.predict(test)
	#print target

	print result
	

def SGDClassifier(x,y):
	sgd = SGDClassifier(loss="log", alpha=0.01, n_iter=200)
	sgd.fit(x,y)

	scores = cross_validation.cross_val_score(sgd, x, y, cv=5)
	result = scores.mean()

	print result

# def readata(filedir, 		     SIZE,  YEAR, MOUTH,  DAY,  EVENT,  CATE, ZIP)
x, y = readata('LA_CRIMESDATA.csv', 'all', 'all', 'all', 'all', 'all', 6, 90012)
MLRClassifier(x,y)
#SGDClassifier(x,y)

