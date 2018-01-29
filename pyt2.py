"""pyt2.py
   
   29/01/18

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Il programma effettua alcune semplici operazioni preliminari
di data-preprocessing al fine di normalizzare i dati e selezionare
solo le features di interesse per la fase di training e di test """



import numpy as np

"""Normalizzazione dei dati"""

dataset = np.loadtxt('Training_set2.txt')
for i in range(1000):
	dataset[i][0] = dataset[i][0]/100
	dataset[i][1] = dataset[i][1]/10
	dataset[i][2] = dataset[i][2]/10
	dataset[i][3] = dataset[i][3]/10
	dataset[i][4] = dataset[i][4]/100000
	dataset[i][5] = dataset[i][5]	

datasetv = np.loadtxt('validation_set.txt')
for i in range(1000):
	datasetv[i][0] = datasetv[i][0]/100
	datasetv[i][1] = datasetv[i][1]/10
	datasetv[i][2] = datasetv[i][2]/10
	datasetv[i][3] = datasetv[i][3]/10
	datasetv[i][4] = datasetv[i][4]/100000
	datasetv[i][5] = datasetv[i][5]	

datasetd = np.loadtxt('data.txt')
for i in range(100000):
	datasetd[i][0] = datasetd[i][0]/100
	datasetd[i][1] = datasetd[i][1]/10
	datasetd[i][2] = datasetd[i][2]/10
	datasetd[i][3] = datasetd[i][3]/10
	datasetd[i][4] = datasetd[i][4]/100000
	datasetd[i][5] = datasetd[i][5]



"""Trasformazione dei datasets in numpy arrays"""

training_data = [None for _ in range(1000)]
for i in range(500):
	training_data[i] = np.array([[dataset[i][0]],[dataset[i][1]],[dataset[i][2]],[dataset[i][3]],[dataset[i][4]],[dataset[i][5]]],dtype=np.float32), np.array([[1]])
for i in range(500,1000):
	training_data[i] = np.array([[dataset[i][0]],[dataset[i][1]],[dataset[i][2]],[dataset[i][3]],[dataset[i][4]],[dataset[i][5]]],dtype=np.float32), np.array([[0]])


validation_data = [None for _ in range(1000)]
for i in range(500):
	validation_data[i] = np.array([[datasetv[i][0]],[datasetv[i][1]],[datasetv[i][2]],[datasetv[i][3]],[datasetv[i][4]],[datasetv[i][5]]],dtype=np.float32), np.array([[1]])
for i in range(500,1000):
	validation_data[i] = np.array([[datasetv[i][0]],[datasetv[i][1]],[datasetv[i][2]],[datasetv[i][3]],[datasetv[i][4]],[datasetv[i][5]]],dtype=np.float32), np.array([[0]])

test_data = [None for _ in range(100000)]
for i in range(100000):
	test_data[i] = np.array([[datasetd[i][0]],[datasetd[i][1]],[datasetd[i][2]],[datasetd[i][3]],[datasetd[i][4]],[datasetd[i][5]]],dtype=np.float32)

