# neural_network-from-numpy
i have built a artificial neural network from numpy . its have so many hyperperameter so we can tune our model as we want . and also i had build a image classifier on there this network and i got 85+ accuracy . its quit good for a ANN even on image dataset .

in this project i have created three python file called network function , mini_batch, and numpy_neuralnetwork 
> 1 . network_function :- it contains the basic neural network function such as weights initilization (i used the he_normal initilization for weights)
and loss function (loss function i used log loss same as tensorflow binarycrossentropy loss) and also the relu , sigmoid , tanh , leakyrelu, derivative of 
relu and tanh and sigmoid and leakyrelu also it contains the forward pass of neural network and beakword part of the network . and one best debug function 
for checking whether your backpropagation is working perfect or not with the help of gradient chack function .

> 2. mini_batches :- this file contains the mini_batch function and the compiler(e.g adam , gd,rmsprop,momentum) and learning rate decay function(continous,descrite) . the mini batch fuction is split the training data into the small packets and increse the training speed.
and in compiler there are 4 compiler is there adam , rmsprop , momentum and gd(gradient descent ) . and learning rate decay function is used 
to decay the rate during train after some epoch . it cotains two type of decay whether it is continous or decrite if its contionus then the learning rate will decay after every epoch and if its descrite then the learning rate will decay after some interval .

> 3. numpy_neuralnetwork :- in this file all training part will done . and it has a fit function that fit the weights to the neural network and with help of evaluate function you can evalute the test data . and from the gradient chack you can chack your back propagation is working fine or not.

IN this neural network have hyperparameter such as :- 

> 1. layer_dims = [] :  this hyperperameter contains the neuron in each layer . note that its also include the input neuron(your data axis =0 is your input neuron e.g x.shape[0]) an output neuron . for e.g layer_dims = [2300,19,15,1] , means your input neuron have 2300 neuron in layer1, 19 neuron in first hiden layer , 15 neuron in seconed hiden layer and one neuron in output layer so its a binary classification .  

>2.  seed= 0 :  it is also a very usefull in case you are stuck in a local optima it set the seed to the rendomly initialized weights. by change the weights  you can escape from an local optima.

>3. keep_prob= 1 : this is used when you are doing the dropout regularization as probabily how much neural you want to cut connections.

>4. learning_rate=0.0075 : 

>5. lambd=1,

>6. regularization=None

>7. print_cost=False

>8. activation_function='relu' 

>9. mini_batch_size=32,

>10. optimizer='gd'

>11. beta1=0.9 

>12. beta2=0.999 

>13. decay_rate=1 

>14. decay=None 

>15. epoch=2000
