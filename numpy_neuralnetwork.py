import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mini_batch as mb
import network_functions as nf
class neural_network():
    def __init__(self,layer_dims = [],seed=0,keep_prob=1,learning_rate = 0.0075,lambd=1,regularization=None,print_cost=False,activation_function='relu',mini_batch_size=32,optimizer='gd',beta1=0.9,beta2=0.999,decay_rate=1,decay=None,epoch=2000):
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_rate = decay_rate
        self.decay = decay
        self.layer_dims =layer_dims
        self.regularization = regularization
        self.print_cost = print_cost
        self.lambd = lambd
        self.activation_function = activation_function
        self.epoch = epoch
        self.keep_prob = keep_prob
        self.seed = seed
        self.mb = mb.mini_batch(self.mini_batch_size,self.optimizer,self.learning_rate,self.beta1,self.beta2,self.decay_rate,self.decay)
        self.nf = nf.neural_function(self.layer_dims,self.seed,self.learning_rate,self.lambd,self.regularization,self.print_cost,self.activation_function,self.keep_prob)
    def fit(self,x,y):
        t = 0
        m = x.shape[1]
        decay = self.decay
        seed = 10
        lr_decay = []
        costs = []
        learning_rate = self.learning_rate
        parameters = self.nf.initialize()
        if self.optimizer == 'gd':
            pass
        elif self.optimizer == 'momentum':
            v = self.mb.initialize_with_momentum(parameters)
        elif self.optimizer == 'rmsprop':
            s = self.mb.initialize_with_rmsprop(parameters)
        elif self.optimizer == 'adam':
            v,s = self.mb.initialize_with_adam(parameters)
        for i in range(self.epoch):
            seed = seed +1
            mini_batches = self.mb.random_suffle(x,y,seed)
            cost = 0
            Accuracy =0

            for mini_batch in mini_batches:

                mini_batch_x,mini_batch_y = mini_batch

                AL,caches = self.nf.L_forward_model(mini_batch_x,parameters)


                c,a= self.nf.compute_cost(AL,mini_batch_y)
                cost +=c
                Accuracy +=a
                grads = self.nf.L_backward_model(AL,mini_batch_y,caches)
                if self.regularization:
                    grads,caches,cost = self.nf.regularize(mini_batch_y,grads,parameters,caches,cost)
                    if self.regularization == 'dropout':
                        grads = self.nf.L_backward_model(AL,mini_batch_y,caches)
                if self.optimizer == 'gd':
                    parameters = self.mb.update_parameters_with_gd(parameters,grads,learning_rate)
                elif self.optimizer == 'momentum':
                    parameters ,v = self.mb.update_parameters_with_momentum(v,parameters,grads,learning_rate)
                elif self.optimizer == 'rmsprop':
                    parameters , s = self.mb.update_parameters_with_rmsprop(s,grads,parameters,learning_rate)
                elif self.optimizer == 'adam':
                    t = t+1
                    parameters , v,s,_,_ = self.mb.update_parameters_with_adam(parameters,grads,v,s,t,learning_rate)

            avg_cost = cost/m
            avg_Accuracy = Accuracy/m
            if self.print_cost and i%100 ==0:
                print("the avg cost after {} epoch is  {} accuracy is  {} %".format(i,avg_cost,avg_Accuracy*100))
                costs.append(avg_cost)
            if decay:
                if decay == 'continuous':

                    learning_rate = self.mb.continuous(learning_rate,i)
                elif decay == 'descrite':
                    learning_rate = self.mb.descrite(learning_rate,i)

            if decay and i%100 == 0:
                print('learning_rate after {} epoch is  {}'.format(i,learning_rate))
                lr_decay.append(learning_rate)
        self.parameters = parameters
        self.costs = costs
        self.lr_decay = lr_decay
    def plot(self,label=None,col='r'):
        if label == 'cost':
            plt.plot(np.squeeze(self.costs),color=col,label=label)
            plt.ylabel('cost')
            plt.xlabel('epoch (per hundreds)')
            plt.show()
        elif label == 'lr_decay':
            plt.plot(np.squeeze(self.lr_decay),color=col,label=label)
            plt.ylabel('learning_rate')
            plt.xlabel('epoch (per hundreds)')
            plt.show()
    def predict(self,x):
        param = self.parameters
        AL,caches = self.nf.L_forward_model(x , param)
        y_pred = (AL > 0.5)
        return y_pred.astype(int)
    def report(self,y_pred,y_test):
        self.nf.model_report(y_pred,y_test)
    def grads_check(self,x,y,epsilon=1e-7,print_msg=False):
        difference = self.nf.grad_check(x,y,epsilon,print_msg=True)

def main():
    x_train = np.load("x_train.npy")
    x_test = np.load("x_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    model = neural_network(layer_dims=[12288,20,7,5,1],keep_prob=0.7,regularization=None,lambd=1,learning_rate=0.01,decay_rate=0.01,decay='descrite',print_cost=True,optimizer='gd',activation_function='relu',epoch=501,seed=1)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    model.report(y_pred,y_test)
    model.plot(label='cost')
    model.plot(label='lr_decay')
if __name__ == "__main__":
    main()
