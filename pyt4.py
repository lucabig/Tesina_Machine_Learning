"""pyt4.py
   
   29/01/18

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Il programma applica 'network2.py' e grafica diversi plot per
verificare l'eventuale presenza di overfitting e per monitorare
le prestazioni della rete neurale sul test_set """



import network2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Inizializzazione della rete neurale """
epoche = 500
net = network2.Network([6,12,1], cost = network2.QuadraticCost)
evaluation_cost, evaluation_accuracy,training_cost,training_accuracy = net.SGD(training_data, epoche, 5,1.0,evaluation_data=validation_data,monitor_evaluation_accuracy = True,monitor_evaluation_cost = True,monitor_training_accuracy = True, monitor_training_cost = True)


"""Plot del costo sul training set"""
plt.plot(np.linspace(1,epoche,epoche),training_cost, color = 'red')
plt.title('training cost')
plt.show()


"""Plot del costo sul validation set"""
plt.plot(np.linspace(1,epoche,epoche),evaluation_cost, color = 'blue')
plt.title('validation cost')
plt.show()


"""Plot dell'accuratezza sul training set"""
plt.plot(np.linspace(1,epoche,epoche),training_accuracy, color ='red')
plt.title('training accuracy')
plt.show()


"""Plot dell'accuratezza sul validation set"""
plt.plot(np.linspace(1,epoche,epoche),evaluation_accuracy, color = 'blue')
plt.title('validation accuracy')
plt.show()


"""Plot del risultato della classificazione"""
outs = net.outputs(validation_data)
outb = net.outputb(validation_data)
plt.hist(outs[0], bins=50, alpha=0.5, label='signal')
plt.hist(outb[0], bins=50, alpha=0.5, label='background')
plt.legend(loc='upper right')
plt.title('Classification')
plt.show()




"""Plot della distribuzione della massa invariante dopo l'applicazione di un taglio"""
out = net.output(test_data)
temp = []
for i in xrange(len(test_data)):
	if out[i] > 0.9985:
		temp.append(i)			
massa_segnale = []
for i in xrange(len(temp)):
	massa_segnale.append(datasetd[temp[i]][6])

plt.hist(massa_segnale, bins=40, alpha=0.5, label='signal')
plt.title('Invariant mass distribution after the cut')
plt.show()



"""Plot della distribuzione della massa invariante prima dell'applicazione del taglio"""
massa = []
for i in xrange(len(test_data)):
	massa.append(datasetd[i][6])

plt.hist(massa, bins=40, alpha=0.5, label='signal')
plt.title('Invariant mass distribution before the cut')
plt.show()




"""Plot delle efficienze di segnale e background in funzione dei valori di taglio impostati"""
TPRs = []
TPRb = []
taglio = np.linspace(0,1,1001)
for i in taglio:
	TPs=[]
	TPb = []
	for x in xrange(len(validation_data)):
		if((net.feedforward(validation_data[x][0])[0][0])>i):
			if(validation_data[x][1][0][0] == 1):
				TPs.append(validation_data[x][0][0][0])
		if((net.feedforward(validation_data[x][0])[0][0])<i):
			if(validation_data[x][1][0][0] == 0):
				TPb.append(validation_data[x][0][0][0])

	TPRs.append(float(len(TPs))/(500))
	TPRb.append(1-(float(len(TPb))/(500)))	

plt.plot(taglio,TPRs, color ='red')
plt.plot(taglio,TPRb, color = 'blue')
plt.title('Background Efficiency vs. Signal Efficiency in function of the cut')
plt.show()
