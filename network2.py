"""network2.py
   
   29/01/18

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Il programma implementa l'algoritmo di apprendimento stochastic gradient descent
su una feedforward neural network. I gradienti sono calcolati per mezzo dell'algoritmo 
di backpropagation. Possono essere impostate due diverse loss functions: la quadratica
e la logistica. """

#### Librerie
import json
import random
import sys
import numpy as np


#### Definizione delle loss functions.

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Fornisce il costo associato all'output della rete ``a`` e 
	all'output desiderato ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Fornisce l'errore 'delta' associato all'ultimo layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Fornisce il costo associato all'output della rete ``a`` e 
	all'output desiderato ``y``.  """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Fornisce l'errore 'delta' associato all'ultimo layer."""
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """La lista ``sizes`` contiene il numero di neuroni nei rispettivi
        layer della rete.  Per esempio, se la lista fosse [2, 3, 1]
        allora si avrebbe una rete a tre layer, con il primo layer
        contenente 2 neuroni, il secondo layer 3 neuroni, e il terzo
        layer 1 neurone.  """
        
	self.num_layers = len(sizes)
        self.sizes = sizes
        self.large_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Inizializza ogni peso estraendo da una distribuzione Gaussiana 
        con media 0 e deviazione standard 1 diviso la radice quadrata del
        numero di pesi connessi allo stesso neurone. I bias sono inizializzati
	mediante una distribuzione Gaussiana a media 0 e deviazione standard 1"""
	
	np.random.seed(1)  
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
	np.random.seed(55)  
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Inizializza ogni peso estraendo da una distribuzione Gaussiana 
        con media 0 e deviazione standard 1. I bias sono inizializzati
	mediante una distribuzione Gaussiana a media 0 e deviazione standard 1. """
	np.random.seed(1) 
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
	np.random.seed(55)  
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Fornisce la risposta della rete se ``a`` rappresenta il dato di input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Addestra la rete neurale uasando l'algoritmo ''mini-batch 
	stochastic gradient descent. ``training_data`` rappresenta un dataset 
	del tipo ``(x, y)`` contenente i dati di input utilizzati per
	l'addestramento e gli output desiderati. Il metodo accetta 
	anche il dataset ``evaluation_data``, utilizzato come validation set.
        Si possono monitorare gli errori commessi e l'accuratezza (in funzione
	del numero di epoche di training) sia sul  validation set, sia sul training set. """
	random.seed(17)
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
	average = 0
	temp = eta
        for j in xrange(epochs):
	    eta = eta-(float(temp-0.001)/(epochs))
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data)
		if (epochs-j)<50:
                	average = average + self.evaluate(evaluation_data)

            print
	Average = 0
	Average = float(average)/(50)
	print "Average over the last 50 outputs: {0}".format(Average)
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Aggiorna i pesi e i bias della rete applicando l'algoritmo 
        gradient descent utilizzando l'algoritmo di backpropagation su 
	un singolo mini batch. ``mini_batch`` rappresenta un dataset del tipo ``(x, y)``,
	``eta`` il learning rate, ``lmbda`` il parametro di regolarizzazione,
	e ``n`` la taglia del training set."""
        
	nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Fornisce un dataset ``(nabla_b, nabla_w)`` che rappresenta il
        gradiente della funzione costo C_x;  ``nabla_b`` e
        ``nabla_w`` sono liste di numpy arrays layer per layer, simili
        a ``self.biases`` e ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward 
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data):
        """Ritorna il numero di input presenti in ``data`` per
	cui la rete neurale fornisce il risultato corretto. """
        
        results = [(self.feedforward(x), y)
                       for (x, y) in data]
 
        return sum(int(np.absolute(x-y)<0.5) for (x, y) in results)

    def total_cost(self, data, lmbda):
        """Ritorna il costo complessivo per il dataset ``data``.  """
        
	cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    
    def output(self, training_data):
        """Ritorna una lista contenente gli output della rete quando in 
	input vengono forniti i dati del test_set """            
	
	t = []
    	for i in range(len(training_data)):
		t.append(self.feedforward(training_data[i])[0][0])
        out = t
	return out   

    
    def outputs(self, training_data):
        """Ritorna una lista contenente gli output della rete quando in 
	input vengono forniti i dati del training_set corrispondenti a eventi di segnale """
        
	t = []
	s = []
    	for i in range(len(training_data)):
		if training_data[i][1][0][0] == 1:
			t.append(self.feedforward(training_data[i][0])[0][0])
			s.append(training_data[i][1][0][0])
        out = [t,s]
	return out


    def outputb(self, training_data):
        """Ritorna una lista contenente gli output della rete quando in 
	input vengono forniti i dati del training_set corrispondenti a eventi di background """
        
	t = []
	s = []
    	for i in range(len(training_data)):
		if training_data[i][1][0][0] ==0:
			t.append(self.feedforward(training_data[i][0])[0][0])
			s.append(training_data[i][1][0][0])
        out = [t,s]
	return out


	
    

    def evaluate(self, test_data):
        """Ritorna il numero di dati per cui la rete fornisce il 
	risultato corretto. """

        test_results = [(self.feedforward(x), y)
                        for (x, y) in test_data]
        return sum(int(np.absolute(x-y)<0.5) for (x, y) in test_results)



#### Funzioni utili
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
