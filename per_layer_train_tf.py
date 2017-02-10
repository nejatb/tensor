import numpy as np
import tensorflow as tf


class MLP:

    def __init__(self, n_input, n_classes, neurons):
        # Parameters
        self.learning_rate = 0.001
        self.training_epochs = 100

        self.batch_size = 100
        self.display_step = 1

        #Network Parameters
        self.n_input = n_input
        self.n_classes = n_classes
        self.neurons = neurons

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Store Layers weight and bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.neurons[0]])),
            'out': tf.Variable(tf.random_normal([self.neurons[0], n_classes]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.neurons[0]])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Construct model
        self.pred = self.multilayer_perceptron()

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)

        #initializing the variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def multilayer_perceptron(self):
        # Hidden Layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.tanh(layer_1)

        # Output layer with linear activation
        out_layer = tf.matmul(layer_1, self.weights['out']) + self.biases['out']
        #layer_2 = tf.nn.softmax(layer_1)
        return out_layer

    def fit(self,batch_x,batch_y):

        for epoch in range(self.training_epochs):
            _,c = self.sess.run([self.optimizer, self.cost], feed_dict = { self.x: batch_x, self.y : batch_y} )

            print("Epoch:{} , cost:{:.9f}".format(epoch,c))

        print("Optimization finished!")

    def predict(self,batch_x,batch_y):
        '''
        correct_prediction = tf.equal(tf.argmax(self.pred,1), tf.argmax(batch_y,1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy", accuracy.eval({self.x: batch_x, self.y : batch_y }, session= self.sess))
        '''
        prediction = tf.argmax(tf.nn.softmax(self.pred),1)

        return prediction.eval(feed_dict={self.x: batch_x}, session= self.sess)

    def get_weight(self,layer):
        return self.weights[layer].eval(session = self.sess)