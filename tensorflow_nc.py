from __future__ import division
import numpy as np
import tensorflow as tf


class MLP:

    def __init__(self, n_input, n_classes, neurons):
        # Parameters
        self.learning_rate = 0.001
	self.reg_constant = 0.01
        self.training_epochs = 100
        self.batch_size = 100
        self.display_step = 1

        #Network Parameters
        self.n_input = n_input
        self.n_classes = n_classes
        self.neurons = neurons

        stdinit = 0.04

        # tf Graph input
        self.x1 = tf.placeholder("float", [None, self.n_input])
	self.x2 = tf.placeholder("float",[None, self.n_input])
	self.y = tf.placeholder("float", [None, self.n_classes])

        # Store Layers weight and bias
        self.weights = {
            'w1': tf.Variable(tf.random_uniform([self.n_input, self.neurons[0]], minval = -stdinit, maxval = stdinit)),
	    'w2': tf.Variable(tf.random_uniform([2*self.neurons[0], self.neurons[0]], minval = -stdinit, maxval = stdinit)),
	    'out': tf.Variable(tf.random_uniform([self.neurons[0], n_classes], minval = -stdinit, maxval = stdinit))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_uniform([self.neurons[0]], minval = -stdinit, maxval = stdinit)),
	    'b2': tf.Variable(tf.random_uniform([self.neurons[0]], minval = -stdinit, maxval = stdinit)),
           'out': tf.Variable(tf.random_uniform([self.n_classes], minval = -stdinit, maxval = stdinit))
        }

        # Construct model
        self.pred = self.multilayer_perceptron()

        # Define regularizer, loss and optimizer
	reg_loss = self.reg_constant * tf.nn.l2_loss(self.weights['w1']) + self.reg_constant* tf.nn.l2_loss(self.weights['out']) 
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = self.y)) + reg_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
        #self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.learning_rate).minimize(self.cost)

        #initializing the variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def multilayer_perceptron(self):
        # Hidden Layer with RELU activation
	layer_x1 = tf.add(tf.matmul(self.x1,self.weights['w1']),self.biases['b1'])
	layer_x1 = tf.nn.relu(layer_x1)

	layer_x2 = tf.add(tf.matmul(self.x2, self.weights['w1']),self.biases['b1'])
	layer_x2 = tf.nn.relu(layer_x2)

	x = tf.concat(1,[layer_x1, layer_x2])
	layer_1 = tf.add(tf.matmul(x,self.weights['w2']),self.biases['b2'])
	layer_1 = tf.nn.relu(layer_1)

        # Output layer with linear activation
        out_layer = tf.matmul(layer_1, self.weights['out']) + self.biases['out']
        #layer_2 = tf.nn.softmax(layer_1)
        return out_layer

    
    def fit(self,batch_x1, batch_x2,batch_y):

        for epoch in range(self.training_epochs):
            _,c = self.sess.run([self.optimizer, self.cost], feed_dict = { self.x1: batch_x1,self.x2:batch_x2, self.y : batch_y} )
		
            print("Epoch:{} , cost:{:.9f}".format(epoch,c))
	     
	    if ( epoch % 10 == 0):
		print("Training")
		self.accuracy(batch_x1, batch_x2,batch_y)
		#print("Testing")
		#self.accuracy(test_x1,test_x2,test_y)
		
	    	
		#print(self.predict(batch_x,batch_y))

        print("Optimization finished!")

    def predict(self,batch_x1, batch_x2):
        '''
        correct_prediction = tf.equal(tf.argmax(self.pred,1), tf.argmax(batch_y,1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy", accuracy.eval({self.x: batch_x, self.y : batch_y }, session= self.sess))
        '''
        prediction = tf.argmax(self.pred,1)

        return prediction.eval(feed_dict={self.x1: batch_x1, self.x2:batch_x2}, session= self.sess)

    def get_weight(self,layer):
        return self.weights[layer].eval(session = self.sess)

    def accuracy(self, batch_x1,batch_x2, batch_y):
	correct_prediction = tf.equal(tf.argmax(self.pred,1), tf.argmax(batch_y,1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy", accuracy.eval({self.x1: batch_x1, self.x2:batch_x2, self.y : batch_y }, session= self.sess))
