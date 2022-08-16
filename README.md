# neural_network-from-numpy
i have built a artificial neural network from numpy . its have so many hyperperameter so we can tune our model as we want . and also i had build a image classifier on there this network and i got 85+ accuracy . its quit good for a ANN even on image dataset .

In this project i have created three python file called network function , mini_batch, and numpy_neuralnetwork 
> 1 . network_function :- it contains the basic neural network function such as weights initilization (i used the he_normal initilization for weights)
and loss function (loss function i used log loss same as tensorflow binarycrossentropy loss) and also the relu , sigmoid , tanh , leakyrelu, derivative of 
relu and tanh and sigmoid and leakyrelu also it contains the forward pass of neural network and beakword part of the network . and one best debug function 
for checking whether your backpropagation is working perfect or not with the help of gradient chack function .

> 2. mini_batches :- this file contains the mini_batch function and the optimizer (e.g adam , gd,rmsprop,momentum) and learning rate decay function(continous,descrite) . the mini batch fuction is split the training data into the small packets and increse the training speed.
and in compiler there are 4 optimizer is there adam , rmsprop , momentum and gd(gradient descent ) . and learning rate decay function is used 
to decay the rate during train after some epoch . it cotains two type of decay whether it is continous or decrite if its contionus then the learning rate will decay after every epoch and if its descrite then the learning rate will decay after some interval .

> 3. numpy_neuralnetwork :- in this file all training part will done . and it has a fit function that fit the weights to the neural network and with help of evaluate function you can evalute the test data . and from the gradient chack you can chack your back propagation is working fine or not.

IN this neural network have hyperparameter such as :- 

> 1. layer_dims = [] :  this hyperparameter contains the neuron in each layer . note that its also include the input neuron(your data axis =0 is your input neuron e.g x.shape[0]) an output neuron . for e.g layer_dims = [2300,19,15,1] , means your input neuron have 2300 neuron in layer1, 19 neuron in first hidden layer , 15 neuron in seconed hidden layer and one neuron in output layer so its a binary classification .  

>2.  seed= 0 :  it is also a very usefull in case you are stuck in a local optima it set the seed to the rendomly initialized weights. by change the weights  you can escape from an local optima.

>3. keep_prob= 1 : this is used when you are doing the dropout regularization as probabily how much neural you want to cut connections.

>4. learning_rate=0.0075 : this is scaler value that is used for to minimize the cost function . the value of learning rate descide that how aggressively
we are going towards the minimum of the cost function . range of learning rate (0.000001 , 1)

>5. lambd=1 : this hyperparameter used in L2 reguralization to control the weight decay. range of lambd is (0.1 , 100) 

>6. regularization = None : this hyperparameter will take the type of regularizer whether it is a 'dropout' or 'l2_regularizer' or None for without regularization training . its very importent to use regularization in neural network to prevent the overfiting or high varience problem . 

>7. print_cost = False : this hyperparameter will take either True or False . True means you want to see the cost of function after every epoch. 

>8. activation_function = 'relu' : this hyperparameter give you the option that which activation function do you want to use in your hidden layer. you 
can your four type of function relu , tanh , sigmoid , leakyrelu you can use according to your prefarence (relu is best option when you don't know which one to use) 

>9. mini_batch_size=32 : this is use to make packets of training data mean in mini batches how many examples do you want in each packets or batch of your training dataset . 32 means we have 32 examples in a batch of training dataset.

>10. optimizer='gd' : this hyperparameters give you a choice to choose one optimizer out four optimizer which is 'gd' ,'adam','rmsprop','momentum'.
this optimizer is allow you to update the perameter .with gd or adam or momentum or rmsprop. its definatly increse the performance of model as well as 
the speed of gradient descent(with the choice of a right optimizer)

>11. beta1=0.9 : this is used in adam ,momentum and rmsprop  optimizer to control the weights average . (keep it default its works well for most of the models)

>12. beta2=0.999 : this will use only in Adam optimizer.(keep it default in most of the case)

>13. decay_rate = 1 : this i a scaler values thats been used to decay the learning rate after every epoch . its better to have small because its decay exponentialy the learning rate. 

>14. decay = None : this hyperperameter have two option to choose the type of learning rate decay whether it is continuous or descrite . continuous will decay the learning rate after every epoch and decrite will decay the learning rate by some interval of epoch. 

>15. epoch = 2000 : its used to iterate the data through the model and the epoch tells you the how many times you are going to iterate the data through the model . by defalt its 2000.

After building this neural network i have tested this model on a cat noncat image classification dataset which contains the image below:

![image](https://github.com/samthakur587/neural_network-from-numpy/blob/main/image/Figure_2.png)

I have trianed the data with few layers and neurons [12288,20,7,5,1] and got the good accuracy even wirh the ANN or small dataset.
Loss of training dataset : 

![image](https://github.com/samthakur587/neural_network-from-numpy/blob/main/image/cost.png)

Learning rate decay because i used the decrite decay type and 1000 epoch interval so for 500 epoch its remains same .

![image](https://github.com/samthakur587/neural_network-from-numpy/blob/main/image/learning_rate%20decay.png)

After training i have elaluted this on test dataset with 50 examples : 

![image](https://github.com/samthakur587/neural_network-from-numpy/blob/main/image/cunfusion%20matrix.png)

And Finally I got the accuracy and recall and precision as shown below : 

![image](https://github.com/samthakur587/neural_network-from-numpy/blob/main/image/Screenshot%20from%202022-08-16%2017-36-33.png)

So my Final prediction on the test images are shown below : 

![image](https://github.com/samthakur587/neural_network-from-numpy/blob/main/image/cats.png)

So as you can see i have got good accuracy and precision, recll value by this nueral network which is implemented by numpy . notice that this neural network i have implemented as binary classification if you want to do multiclass classification you can do the one hot encoding your y_test and the add a softmax layer to it or else you can use sigmoid function with more then one output neuron.

THANK YOU !

# [LICENSE](LICENSE)

# [LinkedIn](https://www.linkedin.com/in/samunder-singh-265508202/)





