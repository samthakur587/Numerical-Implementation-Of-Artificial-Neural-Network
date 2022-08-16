import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class neural_function():
    def __init__(self,layer_dims = [],seed=0,learning_rate = 0.0075,lambd=1,regularization=None,print_cost=False,activation_function='relu',keep_prob=1):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.print_cost = print_cost
        self.activation_function = activation_function
        self.regularization = regularization
        self.lambd = lambd
        self.seed = seed
        self.keep_prob = keep_prob
    def sigmoid(self,z):
        A = 1/(1+np.exp(-z))
        cache = z
        return A,cache
    def relu(self,z):
        A = np.maximum(0,z)
        assert (A.shape == z.shape)
        cache = z
        return A , cache
    def tanh(self,z):
        A = np.tanh(z)
        assert(A.shape == z.shape)
        cache = z
        return A,cache
    def leakyrelu(self,z):
        A = np.maximum(0.01*z,z)
        assert (A.shape == z.shape)
        cache = z
        return A , cache
    def backward_sigmoid(self,dA,cache):
        z = cache
        s = 1/(1+np.exp(-z))
        dz = dA*s*(1-s)
        assert(dz.shape == z.shape)
        return dz
    def backward_relu(self,dA,cache):
        z = cache
        dz = np.array(dA,copy=True)
        dz[z <= 0] = 0
        assert(dz.shape == z.shape)
        return dz
    def backward_leakyrelu(self,dA,cache):
        z = cache
        dt = np.array(dA,copy=True)
        dt[z > 0] = 1
        dt[z <= 0] = 0.01
        dz = dt*dA
        assert(dz.shape == z.shape)
        return dz
    def backward_tanh(self,dA,cache):
        z = cache
        A = np.tanh(z)
        dz = dA*(1-np.power(A,2))
        assert(dz.shape == z.shape)
        return dz
    def dictionary_to_vector(self,parameter):
        count = 0
        for key, value in parameter.items():
            new_vector = np.array(parameter[key]).reshape((-1,1))
            if count==0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1
        return theta
    def vector_to_dictionary(self,theta,layer):
        param = {}
        count = 0
        for l in range(1,len(layer)):
            param['W'+str(l)] = theta[count :count + layer[l]*layer[l-1]].reshape(layer[l],layer[l-1])
            count = count + layer[l]*layer[l-1]
            param['b'+str(l)] = theta[count:count + layer[l]].reshape(layer[l],1)
            count = count + layer[l]
        return param
    def initialize(self):
        parameters = {}
        np.random.seed(self.seed)
        L = len(self.layer_dims)
        for l in range(1,L):
            parameters['W'+str(l)] = np.random.randn(self.layer_dims[l],self.layer_dims[l-1])*np.sqrt(1/self.layer_dims[l-1])
            parameters['b'+str(l)] = np.zeros((self.layer_dims[l],1))
            assert(parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (self.layer_dims[l], 1))

        return parameters
    def linear_forward(self,A,W,b):
            Z = np.dot(W,A) + b
            cache = (A,W,b)
            assert(Z.shape == (W.shape[0], A.shape[1]))
            return Z,cache
    def linear_activation_forward(self,A_prev,W,b,activation):

        if activation == 'relu':

            Z,linear_cache = self.linear_forward(A_prev,W,b)
            A,activation_cache = self.relu(Z)
        elif activation == 'sigmoid':
            Z,linear_cache = self.linear_forward(A_prev,W,b)
            A ,activation_cache = self.sigmoid(Z)
        elif activation == 'tanh':
            Z,linear_cache = self.linear_forward(A_prev,W,b)
            A , activation_cache = self.tanh(Z)
        elif activation == 'leakyrelu':
            Z,linear_cache = self.linear_forward(A_prev,W,b)
            A,activation_cache = self.leakyrelu(Z)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache,activation_cache)
        return A,cache
    def L_forward_model(self,X,parameters):
        A = X
        self.m = X.shape[1]
        caches = []
        L = len(parameters) // 2
        for l in range(1,L):

            A_prev = A
            A,cache = self.linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation=self.activation_function)
            caches.append(cache)
        AL,cache = self.linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation='sigmoid')
        caches.append(cache)
        assert(AL.shape == (1,X.shape[1]))
        return AL,caches
    def compute_cost(self,AL,Y):
        #m = Y.shape[1]
        t = (AL>0.5).astype(int)
        cost = -np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
        cost = np.squeeze(cost)
        accuracy = np.sum(t*Y + (1-t)*(1-Y))
        return cost , accuracy
    def linear_backward(self,dz,cache):
        A_prev,W,b = cache
        dW = (1/self.m)*np.dot(dz,A_prev.T)
        db = (1/self.m)*np.sum(dz,axis=1,keepdims=True)
        dA_prev = np.dot(W.T,dz)
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev,dW,db
    def linear_activatin_backward(self,dA,cache,activation):
        linear_cache,activation_cache = cache
        if activation =='relu':
            dz = self.backward_relu(dA,activation_cache)
            dA_prev,dW,db = self.linear_backward(dz,linear_cache)
        elif activation == 'tanh':
            dz = self.backward_tanh(dA,activation_cache)
            dA_prev,dW,db = self.linear_backward(dz,linear_cache)
        elif activation == 'leakyrelu':
            dz = self.backward_leakyrelu(dA,activation_cache)
            dA_prev,dW,db = self.linear_backward(dz,linear_cache)
        elif activation == 'sigmoid':
            dz = self.backward_sigmoid(dA,activation_cache)
            dA_prev,dW,db = self.linear_backward(dz,linear_cache)

        return dA_prev,dW,db
    def L_backward_model(self,AL,Y,caches):
        self.m = Y.shape[1]
        self.L = len(caches)
        grads = {}
        Y = Y.reshape(AL.shape)
        dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
        current_cache = caches[self.L-1]
        dA_prev_temp,dW_temp,db_temp = self.linear_activatin_backward(dAL,current_cache,activation='sigmoid')
        grads['dA'+str(self.L-1)] = dA_prev_temp
        grads['dW'+str(self.L)] = dW_temp
        grads['db'+str(self.L)] = db_temp
        for l in reversed(range(self.L-1)):
            current_cache = caches[l]
            dA_prev_temp,dW_temp,db_temp = self.linear_activatin_backward(grads['dA' + str(l+1)],current_cache,activation=self.activation_function)
            grads['dA'+str(l)] = dA_prev_temp
            grads['dW'+str(l+1)] = dW_temp
            grads['db'+str(l+1)] = db_temp
        return grads
    def regularize(self,y,grads,params,caches,cost):

        m = y.shape[1]
        L = len(params)//2
        if self.regularization == 'L2_regularization':
            L2_reg = 0
            for l in range(1,len(self.layer_dims)):
                L2_reg += (self.lambd/(2*m))*np.sum(params['W'+str(l)]**2)
                grads['dW' + str(l)] = grads['dW'+str(l)] + (self.lambd/m)*params['W'+str(l)]
            cost += L2_reg
        elif self.regularization == 'dropout':
            for l in reversed(range(1,L)):
                linear_cache,activation_cache = caches[l]
                A,W,b = linear_cache
                d = np.random.rand(A.shape[0],A.shape[1])
                d = (d < self.keep_prob)
                A = A*d
                A /=self.keep_prob
                linear_cache = (A,W,b)
                caches[l] = (linear_cache,activation_cache)
                grads['dA'+str(l)] = grads['dA'+str(l)]*d
                grads['dA'+str(l)] /=self.keep_prob
        return grads,caches,cost
    def predict(self,X):
        param = self.parameters
        AL,caches = self.L_forward_model(X , param)
        y_pred = (AL > 0.5)
        return y_pred.astype(int)
    def grad_check(self,X,Y,epsilon=1e-7,print_msg=False):
        parameter = self.initialize()
        AL , caches = self.L_forward_model(X,parameter)
        grads = self.L_backward_model(AL,Y,caches)
        layer = self.layer_dims
        grad = {}
        for l in range(1,len(layer)):
            grad['dW'+str(l)] = grads['dW'+str(l)]
            grad['db'+str(l)] = grads['db'+str(l)]
        param_val = self.dictionary_to_vector(parameter)
        grd = self.dictionary_to_vector(grad)
        num_parameters = param_val.shape[0]
        j_plus = np.zeros((num_parameters,1))
        j_minus = np.zeros((num_parameters,1))
        gradappx = np.zeros((num_parameters,1))
        for i in range(num_parameters):
            ## for (j+e) ##
            theta_plus = np.copy(param_val)
            theta_plus[i] = theta_plus[i] + epsilon
            AL,cache = self.L_forward_model(X,self.vector_to_dictionary(theta_plus,layer))
            j_plus[i] = self.compute_cost(AL,Y)
            ## for j-e ##
            theta_minus = np.copy(param_val)
            theta_minus[i] = theta_minus[i] - epsilon
            AL ,cache = self.L_forward_model(X,self.vector_to_dictionary(theta_minus,layer))
            j_minus[i] = self.compute_cost(AL,Y)
            ## for computing the gradappx vector##
            gradappx[i] = (j_plus[i] - j_minus[i])/(2*epsilon)
        numerator =  np.linalg.norm(gradappx - grd)
        denominator =  np.linalg.norm(gradappx) + np.linalg.norm(grd)
        difference =    numerator/denominator
        if print_msg:
            if difference > 2e-7:
                print ("\nThere is a mistake in the backward propagation! difference = " + str(difference))
            else:
                print ("\nYour backward propagation works perfectly fine! difference = " + str(difference))

        return difference
    def model_report(self,y_pred , y_test):
        tp=0
        tn=0
        fp=0
        fn=0
        report = {}
        for i in range(y_test.shape[1]):
            if y_pred[0][i] ==1.0 and y_test[0][i] ==1 :

                tp = tp+1
            if y_pred[0][i] ==1.0 and y_test[0][i] ==0 :
                fn = fn+1
            if y_pred[0][i] ==0.0 and y_test[0][i] ==1 :
                fp = fp+1
            if y_pred[0][i] ==0.0 and y_test[0][i] ==0 :
                tn = tn+1

        confusion_matrix = np.array([[tp,fp],
                                    [fn , tn]])
        plt.figure(figsize=(10,10))
        sns.heatmap(confusion_matrix,annot=True,fmt='d')
        plt.xlabel('predicted values')
        plt.ylabel('actual values')
        plt.show()
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        Accuracy = (tp+tn)/(tp+tn+fp+fn)
        f1 = 2*recall*precision/(precision+recall)

        #print("\nconfusion_matrix : \n",confusion_matrix)

        report['precision'] = str(precision*100) + "%"
        report['recall'] = str(recall*100) + "%"
        report['Accuracy'] = str(Accuracy*100)+"%"
        report['f1_score'] = str(f1*100) + "%"
        data = pd.DataFrame(list(report.items()), columns = ['report', 'value'])
        print("\nreport\n",data)
