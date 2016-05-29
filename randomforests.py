from sklearn.ensemble import RandomForestClassifier

def randomForests(train,test):

	dataTarget = [x[0] for x in train]
	dataTrain = [x[1:] for x in train]
	rf = RandomForestClassifier(n_estimators=1500, n_jobs=3)
	rf.fit(dataTrain, dataTarget)
	predictions = rf.predict(test)
	archivo_prediccion = open("/home/Alvarado/Documents/facultad/submissionRF.csv","a")
	archivo_prediccion.write("ImageId,Label\n")
	imageId = 1
	for label in predictions:
		archivo_prediccion.write(str(imageId) + "," + str(label) + "\n")
		imageId += 1
	archivo_prediccion.close()


#Se recomiendo ejecutar de la siguiente forma
#from numpy import genfromtxt, savetxt
#train = genfromtxt(open("/home/Alvarado/Documents/facultad/7506datos/train.csv","rb"), delimiter=",", dtype="i")[1:]
#test = genfromtxt(open("/home/Alvarado/Documents/facultad/7506datos/test.csv","rb"), delimiter=",", dtype="i")[1:]
#randomForests(train, test)