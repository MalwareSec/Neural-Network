from numpy import exp, array, random, dot
import numpy, sys

class NeuralNetwork():

	def __init__(self):
		# Seed the random number generator, so it generates the same numbers
		# every time the program runs.
		random.seed(1)

		# We model a single neuron, with 3 input connections and 1 output connection.
		# We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
		# and mean 0.
		self.synaptic_weights = 2 * random.random((3, 1)) - 1

	# The Sigmoid function, which describes an S shaped curve.
	# We pass the weighted sum of the inputs through this function to
	# normalise them between 0 and 1.
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	# It indicates how confident we are about the existing weight.
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	# We train the neural network through a process of trial and error.
	# Adjusting the synaptic weights each time.
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
			# Pass the training set through our neural network (a single neuron).
			output = self.think(training_set_inputs)

			# Calculate the error (The difference between the desired output
			# and the predicted output).
			error = training_set_outputs - output

			# Multiply the error by the input and again by the gradient of the Sigmoid curve.
			# This means less confident weights are adjusted more.
			# This means inputs, which are zero, do not cause changes to the weights.
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

			# Adjust the weights.
			self.synaptic_weights += adjustment
	# The neural network thinks.
	def think(self, inputs):
		# Pass inputs through our neural network (our single neuron).
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

	def learn_input(self, input_array, index):
		new_array = numpy.append(training_set_inputs[0], training_set_inputs[1])
		new_array = numpy.concatenate((training_set_inputs, input_array))
		print new_array
		return new_array

	def learn_output(self, output_array, index):
		if output_array >= .50:
			output_array = numpy.array([[1]])
		elif output_array < .50:
			output_array = numpy.array([[0]])
		training_set_outputs.T
		new_outputs = numpy.append(training_set_outputs[0], training_set_outputs[1])
		new_outputs = numpy.concatenate((training_set_outputs, output_array))
		print new_outputs
		return new_outputs

if __name__ == "__main__":

	#Intialise a single neuron neural network.
	neural_network = NeuralNetwork()

	print "Random starting synaptic weights: "
	print neural_network.synaptic_weights

	# The training set. We have 4 examples, each consisting of 3 input values
	# and 1 output value.
	training_set_inputs = array([[1, 0.5, 0], [0, 0.5, 1]])
	training_set_outputs = array([[1, 0]]).T

	# Train the neural network using a training set.
	# Do it 10,000 times and make small adjustments each time.
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print "New synaptic weights after training: "
	print neural_network.synaptic_weights

	while True:
		first_input = raw_input("Enter the first input: ")
		sec_input = raw_input("Enter the second input: ")
		third_input = raw_input("Enter the third input: ")
		first_input = float(first_input)
		sec_input = float(sec_input)
		third_input = float(third_input)
		usr_input = numpy.append(first_input, sec_input)
		usr_input = numpy.append(usr_input, third_input)
		input_array = numpy.array([usr_input])
		print neural_network.think(input_array)
		#May have to split these into two seperate functions
		output_num = numpy.array([neural_network.think(input_array[0])])
		resp = raw_input("Was this outcome %f correct? [Y/N]: " % (output_num))
		if resp.lower() == "y" or resp.lower() == "yes":
			neural_network.learn_input(input_array, i)
			neural_network.learn_output(output_num, i)

		elif resp.lower() == "n" or resp.lower() == "no":
			print "[*] Continuing"
