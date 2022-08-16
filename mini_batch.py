import numpy as np
class mini_batch():
    def __init__(self,mini_batch_size=64,optimizer='gd',learning_rate=0.0075,beta1=0.9,beta2=0.99,decay_rate=None,time_interval=None):
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_rate = decay_rate
        self.time_interval = time_interval
    def random_suffle(self,x,y,seed=0):
        np.random.seed(seed)
        mini_batches = []
        self.m = x.shape[1]
        permutation = list(np.random.permutation(self.m))
        x_shuffle = x[:,permutation]
        y_shuffle = y[:,permutation].reshape((1,self.m))
        inc = self.mini_batch_size
        num_minibatches = self.m//inc
        for i in range(num_minibatches):
            mini_batch_x = x_shuffle[:,i*inc:(i+1)*inc]
            mini_batch_y = y_shuffle[:,i*inc:(i+1)*inc]
            mini_batch = (mini_batch_x,mini_batch_y)
            mini_batches.append(mini_batch)
        if(self.m%inc !=0):
            mini_batch_x = x_shuffle[:,num_minibatches*inc:self.m+1]
            mini_batch_y = y_shuffle[:,num_minibatches*inc:self.m+1]
            mini_batch = (mini_batch_x,mini_batch_y)
            mini_batches.append(mini_batch)
        return mini_batches
    def update_parameters_with_gd(self,parameters,grads,learning_rate):
        L = len(parameters)//2
        for l in range(L):
            parameters['W'+str(l+1)] = parameters['W'+str(l+1)] -(learning_rate)*grads['dW'+str(l+1)]
            parameters['b'+str(l+1)] = parameters['b'+str(l+1)] -(learning_rate)*grads['db'+str(l+1)]
        return parameters
    def initialize_with_momentum(self,parameters):
        L = len(parameters)//2
        v = {}
        for l in range(L):
            v['dW'+str(l+1)] = np.zeros((parameters['W'+str(l+1)].shape[0],parameters['W'+str(l+1)].shape[1]))
            v['db'+str(l+1)] = np.zeros((parameters['b'+str(l+1)].shape[0],parameters['b'+str(l+1)].shape[1]))
        return v
    def update_parameters_with_momentum(self,v,parameters,grads,learning_rate):
        L = len(parameters)//2
        beta = self.beta1
        for l in range(L):
            v['dW'+str(l+1)] = beta*v['dW'+str(l+1)] + (1-beta)*grads['dW'+str(l+1)]
            v['db'+str(l+1)] = beta*v['db'+str(l+1)] + (1-beta)*grads['db'+str(l+1)]
            parameters['W'+str(l+1)]  = parameters['W'+str(l+1)] - learning_rate*v['dW'+str(l+1)]
            parameters['b'+str(l+1)]  = parameters['b'+str(l+1)] - learning_rate*v['db'+str(l+1)]
        return parameters,v
    def initialize_with_rmsprop(self,parameters):
        L = len(parameters)//2
        s = {}
        for l in range(L):
            s['dW'+str(l+1)] = np.zeros((parameters['W'+str(l+1)].shape[0],parameters['W'+str(l+1)].shape[1]))
            s['db'+str(l+1)] = np.zeros((parameters['b'+str(l+1)].shape[0],parameters['b'+str(l+1)].shape[1]))
        return s
    def update_parameters_with_rmsprop(self,s,grads,parameters,learning_rate):
        L = len(s)//2
        beta = self.beta1
        E = 1e-8
        for l in range(L):
            s['dW'+str(l+1)] = beta*s['dW'+str(l+1)] + (1-beta)*np.square(grads['dW'+str(l+1)])
            s['db'+str(l+1)] = beta*s['db'+str(l+1)] + (1-beta)*np.square(grads['db'+str(l+1)])
            parameters['W'+str(l+1)]  = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]/(np.sqrt(s['dW'+str(l+1)])+E)
            parameters['b'+str(l+1)]  = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]/(np.sqrt(s['db'+str(l+1)])+E)

        return parameters , s
    def initialize_with_adam(self,parameters):
        L = len(parameters)//2
        v = {}
        s = {}
        for l in range(L):
            v['dW'+str(l+1)] = np.zeros((parameters['W'+str(l+1)].shape[0],parameters['W'+str(l+1)].shape[1]))
            v['db'+str(l+1)] = np.zeros((parameters['b'+str(l+1)].shape[0],parameters['b'+str(l+1)].shape[1]))
            s['dW'+str(l+1)] = np.zeros((parameters['W'+str(l+1)].shape[0],parameters['W'+str(l+1)].shape[1]))
            s['db'+str(l+1)] = np.zeros((parameters['b'+str(l+1)].shape[0],parameters['b'+str(l+1)].shape[1]))
        return v,s
    def update_parameters_with_adam(self,parameters,grads,v,s,t,learning_rate):
        L = len(parameters)//2
        E = 1e-8
        beta1 = self.beta1
        beta2 = self.beta2
        v_correct = {}
        s_correct = {}
        for l in range(L):
            v['dW'+str(l+1)] = beta1*v['dW'+str(l+1)] + (1-beta1)*grads['dW'+str(l+1)]
            v['db'+str(l+1)] = beta1*v['db'+str(l+1)] + (1-beta1)*grads['db'+str(l+1)]
            s['dW'+str(l+1)] = beta2*s['dW'+str(l+1)] + (1-beta2)*np.square(grads['dW'+str(l+1)])
            s['db'+str(l+1)] = beta2*s['db'+str(l+1)] + (1-beta2)*np.square(grads['db'+str(l+1)])
            v_correct['dW'+str(l+1)] = v['dW'+str(l+1)]/(1-beta1**t)
            v_correct['db'+str(l+1)] = v['db'+str(l+1)]/(1-beta1**t)
            s_correct['dW'+str(l+1)] = s['dW'+str(l+1)]/(1-beta2**t)
            s_correct['db'+str(l+1)] = s['db'+str(l+1)]/(1-beta2**t)
            parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*v_correct['dW'+str(l+1)]/(np.sqrt(s_correct['dW'+str(l+1)])+E)
            parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*v_correct['db'+str(l+1)]/(np.sqrt(s_correct['db'+str(l+1)])+E)
        return parameters,v,s,v_correct,s_correct
    def continuous(self,learning_rate0,num_epoch):
        learning_rate = learning_rate0/(1 + self.decay_rate*num_epoch)
        return learning_rate
    def descrite(self,learning_rate0,num_epoch,time_interval=1000):
        learning_rate = learning_rate0/(1+ self.decay_rate*(num_epoch//time_interval))
        return learning_rate
