# ML-Practice
Some basic ML implementations RAW for just fun.

1. Perceptron on AND Table 
2. Neural Net Implementation from Scratch on MNIST

# NEURAL NETWORK INITIALIZE

N_Net = NeuralNetwork([number_of_hidden_layers, number_of_input_layer_units, number_of_hidden_layer_units, number_of_output_layer_units, number_of_samples]) 

N_Net.structure_define()

N_Net.forward_propagation(X_train)

loss = N_Net.calculate_network_loss(y_train)

N_Net.backward_propagation(X_train, y_train)
