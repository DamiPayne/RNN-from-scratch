##############################################################################
##                     Written by Damilola Payne                            ##
##    Creating a recurrent neural network (RNN) in python with only Numpy   ##
##             Three layer RNN to handle sequential data                    ##
##############################################################################

# Firstly how do Recurrent Neural Nets (RNN) work?
# To understand that you need to understand how information is structured.
# When we you sing a song you not only remember the words but the relationship
# each word has to the previous word.
# If someone was to ask you to sing the song backwards although you remember
# all the words the sequence in which they occur has changed so you cannot sing the song.
# We call this type of information sequential data.
# Regular neural networks are feed-forward meaning that they expect fiXed
# sized inputs and outputs, where the inputs have no relation to the outputs.
# The lack of a relationship means that information only moves in one direction (feed-forward)
# In order to model this type of information we need a neural network which
# can understand sequences, this is why we use RNN's.
# RNN's at every hidden layer not only incorporates the information from the previous
# layer but also the information from the previous time-step recursively.

# So lets write it!
# The only two things we need is 'Numpy' to do our maths and 'Copy' to copy data
# This RNN is going to predict binary sums

import copy
import numpy as np
np.random.seed(0)

# First we need a sigmoid function


def sigmoid(x):
    '''converts values into a probability'''
    output = 1 / (1 + np.exp(-x))
    return output

# Now we need to get the derivative of the sigmoid


def sigmoid_output_to_derivative(output):
    '''calculates the gradient of our sigmoid which is used to find our error)'''
    return output * (1 - output)

# training dataset generation
interger_to_binary = {}  # This is a lookup table that changes integers to binaries
binary_dimensions = 8

largest_number = pow(2, binary_dimensions)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    interger_to_binary[i] = binary[i]

#define input variables
alpha = 0.1
input_dimensions = 2
hidden_dimensions = 16
output_dimensions = 1


# Initialise RNN weights to adjust values
# Synapse 0 connects of the input layer to the hidden layer
# so it has two rows and 16 columns.
synapse_0 = 2 * np.random.random((input_dimensions, hidden_dimensions)) - 1

# Synapse 1 connects the hidden layer to the output layer
# so it has 16 rows and one column.
synapse_1 = 2 * np.random.random((hidden_dimensions, output_dimensions)) - 1

# Synapse h is where the magic of the RNN happens
# it connects the hidden layer in the previous time-step and next time-step
# to the hidden layer in the current time-step
# So it has 16 rows and 16 columns
synapse_h = 2 * np.random.random((hidden_dimensions, hidden_dimensions)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# Training logic
for j in range(10000):

    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number / 2)  # int version
    a = interger_to_binary[a_int]  # binary encoding

    b_int = np.random.randint(largest_number / 2)  # int version
    b = interger_to_binary[b_int]  # binary encoding

    # True answer
    c_int = a_int + b_int
    c = interger_to_binary[c_int]

    #where we will store the binary encoded guess
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dimensions))

    # moving along the positions in the binary encoding
    for position in range(binary_dimensions):

        # generate input and output
        X = np.array([[a[binary_dimensions - position - 1], b[binary_dimensions - position - 1]]])
        y = np.array([[c[binary_dimensions - position - 1]]]).T

        # hidden layer 1
        # is passed through the sigmoid functions as a addition of the layers
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # output layer
        # passes the output from our layers through the sigmoid function
        # to give a prediction
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # calculate the amount you are away from the true value (the error)
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])  # abs is the absolute value (square root of sum)

        # decode estimate so we can print it out
        d[binary_dimensions - position - 1] = np.round(layer_2[0][0])

        #store hidden layer so it can be used it in the next time-step
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dimensions)

    # This is the back propagation (see BPN from scratch for more info)
    for position in range(binary_dimensions):

        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-position - 1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # This information is used to update the new weight update functions
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # print out progress
    if(j % 1000 == 0):
        print("Model Error:" + str(overallError))
        print("Prediction Value:" + str(d))
        print("True Value:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")
