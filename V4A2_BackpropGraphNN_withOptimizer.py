#!/usr/bin/env python
# UA6_1_BackpropGraphNN.py
import numpy as np

# ----------------------------------------------------------------------------------------- 
# Layer: Class for a Neuron Layer  
# ----------------------------------------------------------------------------------------- 
class Layer:
    def __init__(self,l,name,parent,M,h='tanh',c=1.0,init_method='uniform_0'):
        """
        constructor of a neuron layer
        :param l: layer index
        :param name: name of the layer
        :param parent: parent object of the neuron layer (usually a GraphNueralNetwork)
        :param h: activation function (as string like "tanh","sigmoid","linear","relu","lrelu_%d","softmax")
        :param c: gain factor (all activation functions will be multiply firing rates by c)
        :param init_method: initialization method for bias weights (choose between 'zero','uniform_%r','normal_%sig')
        """
        # set parameters
        self.l=l                        # index of layer
        self.name=name                  # name of the layer
        self.parent=parent              # reference to parent (usually GraphNeuralNetwork)
        self.M=M                        # layer size (number of neurons)
        self.h=h                        # activation function (as string like "tanh","sigmoid","linear","relu","lrelu_%d")
        self.c=c                        # gain factor of the layer
        self.init_method=init_method    # set initialization method
        # allocate state variables
        par=self.parent.par
        self.b=np.zeros(M,par['wtype']) # bias weights of layer
        self.a=np.zeros(M,par['stype']) # dendritic/activation potentials of layer
        self.z=np.zeros(M,par['stype']) # firing rates/outputs of layer
        self.alpha=np.zeros(M,par['stype']) # error potentials of layer
        self.delta=np.zeros(M,par['stype']) # error signals of layer
        self.G=np.zeros(M,par['wtype']) # Gradient for bias weights
        # allocate connection information
        self.pre_connections=[]         # preceding connections
        self.succ_connections=[]        # succeeding connections

    def initBiasWeights(self,init_method=None):   # initialize bias weights
        if init_method is None: init_method=self.init_method
        if init_method in ['zero','uniform_0']:
            self.b[:]=np.zeros(self.M)
        else:
            init_method_components=init_method.split('_')   # split at '_'
            error_msg="init_method="+str(init_method)+" should be either 'zero' or 'uniform_%r or normal_%sig'"
            assert len(init_method_components)==2,error_msg+"two components separated by '_' required"
            assert init_method_components[0] in ['uniform','normal'], error_msg+"unknown first component"
            if init_method_component[0] in ['normal']:
                sig=float(init_method_component[1])
                self.b[:]=np.random.normal(0.0,sig,self.W.shape)  # create bias weights according to a normal distribution with standard deviation sig
            else:
                r=float(init_method_component[1])
                self.b[:]=np.random.uniform(-r,r,self.W.shape)    # create bias weights according to a uniform distribution in [-r;r]

    def forward(self,pre_connections): # compute forward pass given source connections pre_connections=[connection1, connection2, ...] (assumes that all preceding layers are already updated)
        # (i) get dendritic potentials a from inputs of preceding layers according to (6.1) in Satz 6.1
        self.a[:]=self.b     # initialize dendritic potentials a with bias b
        for conn in pre_connections:
            assert conn.layer_target==self,"connection "+conn.name+" does not target to layer "+self.name
            conn.propagate_forward()    # add input W*z from source layer to a according to (6.1) in Satz 6.1
        # (ii) compute firing rates by applying activation function h (for details see (6.2) in Satz 6.1 and in SB4, above Kontrollaufgabe 4.4 on page 29)
        h_components=self.h.split('_')   # split at '_'
        assert h_components[0] in ['tanh','sigmoid','linear','relu','lrelu','softmax'],"Unknown activation function h_components[0]="+str(h_components[0])
        if h_components[0]=='tanh': self.z[:]=np.tanh(self.a)                            # tangens hyperbolicus
        elif h_components[0]=='sigmoid': self.z[:]=np.divide(1.0,1+np.exp(-self.a))      # logistic sigmoid function
        elif h_components[0]=='linear': self.z[:]=self.a                                 # linear function
        elif h_components[0]=='relu':                                                    # ReLU (rectifying) activation function
            self.z[:]=self.a
            self.z[self.z<0]=0
        elif h_components[0]=='lrelu':                                                   # leaky ReLU
            self.z[:]=self.a
            self.z[self.z<0]=float(h_components[1])*self.z[self.z<0]
        elif h_components[0]=='softmax':                                                 # Softmax function (see Appendix D.3 in SB3)
            e_a = np.exp(self.a-np.max(self.a))
            self.z[:]=e_a/e_a.sum()
        else:
            assert 0,"Unknown activation function h="+str(self.h)
        self.z[:]*=self.c         # multiply firing rates with gain factor c (default c=1.0)

    def backward(self,succ_connections): # compute backward pass given target connections succ_connections=[connection1, connection2, ...] (assumes that all succeeding layers are already updated)
        # (i) get error potentials alpha from inputs of preceding layers through the transposed weight matrices according to (6.4) in Satz 6.1
        self.alpha[:]=np.zeros(self.M)   # initialize error potentials alpha with 0
        for conn in succ_connections:
            assert conn.layer_source==self,"connection "+conn.name+" does not have source layer "+self.name
            conn.propagate_backward()    # add input W.T*delta from target layer to alpha according to (6.4) in Satz 6.1
        # (ii) compute error signals delta by multiplying with derivative of activation function h' (for details see (6.5) in Satz 6.1 and (6.12)-(6.19) below Satz 6.1)
        h_components=self.h.split('_')   # split at '_'
        assert h_components[0] in ['tanh','sigmoid','linear','relu','lrelu','softmax'],"Unknown activation function h="+str(self.h)
        if h_components[0] in ['tanh','sigmoid','linear','relu','lrelu']:  # simple activation functions?
            if   h_components[0]=='tanh'   : h_=1.0-np.multiply(self.z,self.z)   # h' for tangens hyperbolicus, see (6.15)
            elif h_components[0]=='sigmoid': h_=np.multiply(self.z,1.0-self.z)   # multiply with h' for logistic sigmoid function, see (6.16)
            elif h_components[0]=='linear' : h_=self.delta[:]=1                  # h' for linear function, see (6.19)
            elif h_components[0]=='relu':                                        # h' for ReLU (rectifying) activation function, see (6.18)
                h_=np.ones(self.a.shape,self.a.dtype)
                h_[self.a<0]=0
                h_[self.a==0]=0.5
            elif h_components[0]=='lrelu':                                       # h' for leaky ReLU, see (6.18)
                h_=np.ones(self.a.shape,self.a.dtype)
                h_[self.a<0]=float(h_components[1])
                h_[self.a==0]=0.5*(1.0+float(h_components[1]))
            self.delta[:]=np.multiply(h_,self.alpha)                             # multiply h' and alpha according to (6.5) with (6.14)
        elif h_components[0] in ['softmax']:                                     # multidimensional activation functions?
            if h_components[0]=='softmax':                                       # Dh for Softmax function, see (6.12)
                Jh = np.diag(self.z)-np,outer(self.z,self.z)                     # Jacobian of Softmax (6.12)
            self.delta[:]=np.dot(Jh,self.alpha)                                  # multiply Jacobian Jh with alpha according to (6.5)
        else:
            assert 0,"Error: Invalid activation function h="+str(self.h)
        self.delta[:]*=self.c  # multiply error signals with gain factor c (default c=1.0)
        
    def setBiasGradient(self, flagAccumulate=0):                  # compute gradient for bias weights according to (6.7) in Satz 6.1
        if flagAccumulate<1: self.G[:]=0                          # reset bias gradient to zeros? (e.g., at the beginning of a new learning step)
        if self.l!=0: self.G+=self.delta                          # set (or accumulate over minibatch) current bias gradient according to (6.7) in Satz 6.1 (not required for input layer l=0)

    def getWeights(self): return [self.b]                # get list of all weight matrixes (here only bias vector b)
    def getGradients(self): return [self.G]              # get list of corresponding gradients (here only bias gradient vector G)
        
# ----------------------------------------------------------------------------------------- 
# Connection: Class for a Dense Connection between two Neuron Layers    
# ----------------------------------------------------------------------------------------- 
class Connection:
    def __init__(self,name,parent,layer_target,layer_source,init_method='xavier'):
        self.name=name                  # name of the connection 
        self.parent=parent              # reference to parent (usually GraphNeuralNetwork)
        self.layer_target=layer_target  # reference to target layer
        self.layer_source=layer_source  # reference to source layer
        self.init_method=init_method    # weight initialization method (e.g., 'xavier_uniform/normal', 'he_uniform/normal', 'uniform_%r', 'normal_%sig')
        self.W=np.zeros((layer_target.M,layer_source.M),parent.par['wtype'])  # allocate memory for weights
        self.G=np.zeros((layer_target.M,layer_source.M),parent.par['wtype'])  # allocate memory for gradient matrix of weights

    def initWeights(self,init_method=None):
        if init_method is None: init_method=self.init_method
        init_method_components=init_method.split('_')   # split at '_'
        error_msg="init_method="+str(init_method)+" should be either 'xavier_uniform/normal', 'he_uniform/normal', 'normal_%sig', 'uniform_%r':"
        assert len(init_method_components)==2,error_msg+"two components separated by '_' required"
        assert init_method_components[0] in ['xavier','he','uniform','normal'], error_msg+"unknown first component"
        if   init_method_components[0]=='xavier': sig=np.sqrt(2.0/(self.layer_target.M+self.layer_source.M))  # see (6.48) in Satz 6.4 (Xavier-Initialization) 
        elif init_method_components[0]=='he'    : sig=np.sqrt(4.0/(self.layer_target.M+self.layer_source.M))  # see (6.64) in Satz 6.7 (He-Initialization)
        elif init_method_components[0]=='normal': sig=float(init_method_components[1])
        if init_method_components[0] in ['normal'] or init_method_components[1] in ['normal']:
            self.W[:,:]=np.random.normal(0.0,sig,self.W.shape)        # create weights according to a normal distribution with mean 0 and standard deviation sig 
            self.W[:,:]=np.random.normal(0.0,sig,self.W.shape)        # create weights according to a normal distribution with mean 0 and standard deviation sig 
        else:
            if init_method_components[0]=='uniform': r=float(init_method_components[1])
            else: r=np.sqrt(3.0)*sig
            self.W[:,:]=np.random.uniform(-r,r,self.W.shape)          # create weights according to a uniform distribution in the interval [-r;r]  

    def propagate_forward(self):
        self.layer_target.a+=np.dot(self.W,self.layer_source.z)       # forward propagation contribution W^(l,l_)*z^(l_) according to (6.1) in Satz 6.1 for a dense connection
            
    def propagate_backward(self):
        self.layer_source.alpha+=np.dot(self.W.T,self.layer_target.delta)  # backward propagation contribution W^(l,l_).T*delta^(l) according to (6.4) in Satz 6.1 for a dense connection

    def setGradient(self, flagAccumulate=0):             # compute gradient for synaptic weights according to (6.6) in Satz 6.1
        if flagAccumulate<1: self.G[:,:]=0               # reset gradient to zeros? (e.g., at the beginning of a new learning step)
        self.G+=np.outer(self.layer_target.delta,self.layer_source.z) # set (or accumulate over minibatch) current gradient according to (6.6) in Satz 6.1

    def getWeights(self): return [self.W]                # get list of all weight matrixes (here only W)
    def getGradients(self): return [self.G]              # get list of corresponding gradients (here only G)
        
# ------------------------------------------------------------------------------------------------- 
# Optimizer: Class for an Optimizer doing the weight update (here by stochastic gradient descent)     
# ------------------------------------------------------------------------------------------------- 
class Optimizer:
    def __init__(self,parent):
        self.parent=parent          # reference to GraphNeuralNetwork
        self.init()

    def init(self):                 # initialize optimizer before learning
        par=self.parent.par                 # get parameter dict of parent
        assert par['optimizer'] in ['SGD','ADAM','MOMENTUM'],"Unsupported Optimizer '"+par['optimizer']+"', currently only SGD, MOMENTUM, and ADAM are supported!"
        self.weights,self.gradients=[],[]   # initialize lists of numpy arrays for the weight and gradients of the network
        for l in self.parent.layers:
            self.weights+=l.getWeights()           # append references to bias weights of layer
            self.gradients+=l.getGradients()       # append references to bias gradients of layer
        for c in self.parent.connections:
            self.weights+=c.getWeights()           # append references to weights of connections
            self.gradients+=c.getGradients()       # append references to gradients of connections
        if par['optimizer'] in ['ADAM','MOMENTUM']:
            self.m=[np.zeros(w.shape,dtype=w.dtype) for w in self.weights]   # generate momentum array for each weight/bias array 
        if par['optimizer'] in ['ADAM'           ]:
            self.v=[np.zeros(w.shape,dtype=w.dtype) for w in self.weights]   # generate variance array for each weight/bias array 

    def update(self):                              # update model parameters
        assert len(self.weights)==len(self.gradients),"there must be the same numbers of weights and gradient arrays!"
        eta=self.parent.lr_scheduler.eta           # get current learning rate
        par=self.parent.par                        # get reference to parameter dict
        assert par['optimizer'] in ['SGD','ADAM','MOMENTUM'],"Unsupported Optimizer '"+par['optimizer']+"', currently only SGD, MOMENTUM, and ADAM are supported!"
        if par['optimizer']=='SGD':
            for i in range(len(self.weights)):
                self.weights[i]+=-eta*self.gradients[i] # simple gradient descent with learning rate eta
        elif par['optimizer']=='MOMENTUM':
            for i in range(len(self.weights)):
                w,g,m,beta=self.weights[i].flat,self.gradients[i].flat,self.m[i].flat,par['opt_beta'] # get references to relevant arrays and parameters 
                m[:]=beta*m[:]+eta*g[:]                 # update momentum, see (6.108)
                w[:]-=m[:]                              # update weights, see (6.109)
        elif par['optimizer']=='ADAM':
            tau,beta1,beta2,eps=self.parent.epoch+1,par['opt_beta1'],par['opt_beta2'],par['opt_eps'] # get relevant parameters 
            for i in range(len(self.weights)):
                w,g,m,v=self.weights[i].flat,self.gradients[i].flat,self.m[i].flat,self.v[i].flat  # get references to relevant arrays
                m[:]=beta1*m[:]+(1.0-beta1)*g[:]        # update first moment, see (6.124)
                v[:]=beta2*v[:]+(1.0-beta2)*np.multiply(g[:],g[:]) # update second moment, see (6.125)
                m_hat=np.divide(m[:],1.0-beta1**tau)    # corrected first moment (6.126)
                v_hat=np.divide(v[:],1.0-beta2**tau)    # corrected second moment (6.126)
                w[:]-=eta*np.divide(m_hat[:],np.sqrt(v[:])+eps) # weight update (6.127)
            
# ------------------------------------------------------------------------------------------------- 
# LearningRateScheduler: Class for an schedulzing a learning rate eta for an Optimizer     
# ------------------------------------------------------------------------------------------------- 
class LearningRateScheduler_SimpleDecay:
    def __init__(self,parent):
        self.parent=parent          # reference to GraphNeuralNetwork
        self.eta=parent.par['eta0'] # initialize with default learning rate eta0
        self.init()

    def init(self):
        pass    # here nothing to do 

    def update(self):
        par=self.parent.par        # get global parameter dict
        epoch=self.parent.epoch    # current epoch
        self.eta=par['eta0']/(1.0+par['eta_fade']*float(epoch))  # compute current learning rate (see SB4, Uebungsaufgabe 4.5d)

# ----------------------------------------------------------------------------------------- 
# GraphNeuralNetwork: Feedforward Graph Neural Network with dense layers   
# ----------------------------------------------------------------------------------------- 
class GraphNeuralNetwork: 
    def __init__(self,par={'eta0':0.01,\
                           'eta_fade':0.0,\
                           'lmbda':0.0,\
                           'batch_size':1,\
                           'maxEpochs':100,\
                           'debug':0,\
                           'flagClassify':1,\
                           'modeClassify':'single',\
                           'threshClassify':0.5,\
                           'stype':'float',\
                           'wtype':'float'}): # constructor with parameter dict
        self.par=par                    # parameter dict
        self.L=-1                       # numer of layers (without the input layer)
        self.M=[]                       # list of layer sizes (or nodes)
        self.layers=[]                  # list of neuron layers
        self.connections=[]             # list of connections
        self.dict_layers={}             # empty dict of associations name->layer
        self.dict_connections={}        # empty dict of associations name->connection
        self.N_train=None               # number of training data vectors not yet determined
        self.N_val=None                 # number of validation data vectors not yet determined
        self.D=None                     # number of input units (=dimension of data) not yet determined
        self.K=None                     # number of output units (=dimension of target vectors) not yet determined
        self.X_train=None               # training data not yet defined
        self.T_train=None               # training target values not yet defined
        self.X_val=None                 # validation data not yet defined
        self.T_val=None                 # validation target values not yet defined
        self.flagRecompile=1            # if >0 then graph has been modified before last call to compile() and needs to be recompiled
        self.epoch=0                    # current learning epoch
        self.tau=0                      # current learning step 

    def addLayers(self,M_new,names_new=None,h_new='tanh',c_new=1.0,flagReset=0):
        """ 
        add new layers to the network 
        :param M_new: Either single integer (add one new layer) or list [M_new1,M_new2,...] (for adding several new layers)
        :param names_new: list of names for each new layer; if None then name=layer index as a string for each layer
        :param h_new: activation function for the new layers; either string or list of strings (e.g., 'tanh','sigmoid','linear','relu','lrelu','softmax')
        :param c_new: gain factors for new layers; either float (same gain factor for all layers) or list of floats 
        :param flagReset: if >0 then delete old layers before inserting new layers
        """
        self.flagRecompile=1           # change of network structure needs re-compile
        if flagReset>0:                # delete old layers?
            self.L=-1
            self.M=[]
            self.layers=[]
            self.connections=[]
            self.dict_layers={}
            self.dict_connections={}
        if np.isscalar(M_new): M_new=[M_new]  # cast M_new as a list of necessary
        assert np.isscalar(h_new) or len(M_new)==len(h_new),"length of activation function list h_new="+str(h_new)+" must equal length of layer size list M_new="+str(M_new)
        assert np.isscalar(c_new) or len(M_new)==len(c_new),"length of activation function list h_new="+str(h_new)+" must equal length of layer size list M_new="+str(M_new)
        for i in range(len(M_new)):
            l=self.L+1                             # index of new layer
            Ml=M_new[i]                            # size of new layer
            if names_new is None: name=str(l)      # default name?
            else: name=names_new[i]                # or given name for new layer
            assert not name in self.dict_layers.keys(),"Please use unique neuron layer names: layer name "+str(layer_new)+" is already used!"            
            if np.isscalar(h_new): h=h_new         # same activation function for all new layers? 
            else: h=h_new[i]                       # or individual activation functions?
            if np.isscalar(c_new): c=c_new         # same gain factor for all new layers?
            else: c=c_new[i]                       # or individual gain factors?
            layer_new=Layer(l,name,self,Ml,h,c)    # construct new layer
            self.layers.append(layer_new)          # add new layer to network
            self.M.append(Ml)                      # add layer size
            self.L=self.L+1                        # increase layer number
            self.dict_layers[name]=layer_new       # add entry to dict
        
    def addConnection(self,l,l_,name=None,init_method='xavier_uniform',flagNames=0):
        """
        add new connections from layer l_ to layer l
        :param name: name of the connection 
        :param l : index of the target layer of the connection; if flagNames>0 then l is interpreted as the name of the target layer
        :param l_: index of the source layer of the connection; if flagNames>0 then l_ is interpreted as the name of the source layer
        :param init_method: init method for weight initialization (choose from 'xavier_uniform/normal', 'he_uniform/normal', 'normal_%sig', 'uniform_%r')
        :param flagNames: if >0 then l and l_ are interpreted as name of the target and source layers
        """
        self.flagRecompile=1   # change of network structure needs re-compile
        if flagNames>0:
            assert l  in self.dict_layers.values(),"target layer with name l ="+str(l) +" is unknown!"
            assert l_ in self.dict_layers.values(),"source layer with name l_="+str(l_)+" is unknown!"
            l,l_=int(self.dict_layers[l]),int(self.dict_layers[l_])
        assert l >=0 and l <=self.L,"invalid target layer index l ="+str(l) +", where L="+str(L)
        assert l_>=0 and l_<=self.L,"invalid source layer index l_="+str(l_)+", where L="+str(L)
        assert l>l_,"only feed-forward connections with l>l_ allowed, but you tried to insert a connection from layer l="+str(l)+" to layer l_="+str(l_)
        if name is None: name=str((l,l_))
        assert not name in self.dict_connections.keys(),"Please use unique connection names! A Connection with name="+str(name)+" is already used!"
        conn_new = Connection(name,self,self.layers[l],self.layers[l_],init_method)
        self.layers[l].pre_connections.append(conn_new)   # append new connection to the preceding connections of target layer l
        self.layers[l_].succ_connections.append(conn_new) # append new connection to the succeeding connections of source layer l_
        self.connections.append(conn_new)
        self.dict_connections[name]=conn_new

    def compile(self,flagInit=1): # compile network model, initialize weights, set lr_scheduler and optimizer
        par=self.par
        # (i) compile network
        self.D=self.M[0]   # number of input units
        self.K=self.M[-1]  # number of output units
        pass   # currently nothing to compile; network is build already during definition
        # (ii) # set learning rate scheduler
        assert par['lr_scheduler'] in ['simple_decay'], "par[lr_scheduler]="+str(par['lr_schedulre'])+" is currently not supported!"
        if par['lr_scheduler']=='simple_decay':
            self.lr_scheduler=LearningRateScheduler_SimpleDecay(parent=self)
        # (iii) # set optimizer
        #assert par['optimizer'] in ['SGD'], "par[optimizer]="+str(par['optimizer'])+" is currently not supported!"
        #if par['optimizer']=='SGD':
        self.optimizer=Optimizer(parent=self)
        # (iv) initialize model
        if flagInit:
            self.initWeights()
            self.lr_scheduler.init()
            self.optimizer.init()
        
    def initWeights(self):              # initialize all weights and biases (with random numbers)
        for l in self.layers: l.initBiasWeights()
        for c in self.connections: c.initWeights()

    def getWeights(self):     # get list of references to all weight arrays (=parameter arrays) of the model (e.g., to save, or to update, etc.) 
        weights=[] # initialize lists of numpy arrays for the weights of the network
        for l in self.layers:
            weights+=l.getWeights() # get weights of layer and append to list of model weights
        for c in self.connections:
            weights+=c.getWeights() # get weights of connection and append to list of model weights
        return weights

    def setWeights(self,weights_new):  # set model weights with given list of model weights (e.g., as loaded from file); in-place operations!
        weights_old=self.getWeights()  # get list of old weight arrays of the model to be overwritten (in place)
        assert len(weights_new)==len(weights_old),"Weights to be set must have same number of arrays as model weights list!"
        for i in range(len(weights_old)):
            weights_old[i].flat[:]=weights_new[i].flat    # copy new weight values (in place) to corresponding weight array of the model

    def fun_initdelta_differror(y,t): return y-t  # simple difference errors: delta^L = y-t  (default init of error signals delta^L in output layer for standard cases; see (6.23))

    def forward_pass(self,x):  # compute forward pass for input vector x (that is, (6.1)-(6.2) in Satz 6.1)
        self.layers[0].z[:]=x     # set rates of input layer
        self.layers[0].a[:]=x     # set potentials of input layer (not really required)
        for l in self.layers[1:]: # loop over all non-input layers l
            l.forward(l.pre_connections) # compute forward pass for layer l according to (6.1)-(6.2) in Satz 6.1
        
    def backward_pass(self,t,fun_initdelta):  # compute backward pass for target vector t (that is, (6.3)-(6.5) in Satz 6.1)
        self.layers[-1].delta[:]=fun_initdelta(self.layers[-1].z,t)       # set error signals delta in output layer according to (6.3)
        self.layers[-1].alpha[:]=self.layers[-1].delta/self.layers[-1].c  # set error potentials in output layer (not really required)
        for l in list(reversed(self.layers))[1:]: # loop over all non-output layers l in reversed order (from output to input)
            l.backward(l.succ_connections)  # compute backward pass for layer l according to (6.4)-(6.5)

    def setGradients(self,flagAccumulate=0):     # set gradients according to (6.6)-(6.7) in Satz 6.1 for weight update
        for l in self.layers: l.setBiasGradient(flagAccumulate)   # ... for bias weights
        for c in self.connections: c.setGradient(flagAccumulate)  # ... for synaptic weights of connections

    def getGradients(self):   # return all gradient arrays of the network
        gradients=[]                    # initialize list of numpy arrays for the gradients of the network
        for l in self.layers:
            gradients+=l.getGradients() # get gradients of layer and append to list of model gradients
        for c in self.connections:
            gradients+=c.getGradients() # get gradients of connection and append to list of model gradients
        return gradients
        
    def fit_batch(self,X,T,fun_initdelta=fun_initdelta_differror,flagOptimize=1):   # do weight update over a minibatch according to Satz 6.1
        assert len(X)==len(T),"X and T must have same length, but X.shape="+str(X.shape)+" and T.shape="+str(T.shape)
        N=len(X)                                      # batch size
        Y=np.zeros((N,self.K),self.par['stype'])      # allocate memory for outputs
        flagAccumulate=0                              # reset gradients to zeros for first data vector n=0
        for n in range(len(X)):                       # loop over minibatch
            self.forward_pass(X[n])                   # do forward pass (6.1)-(6.2)
            Y[n,:]=self.layers[-1].z                  # get and store output for x[n]
            self.backward_pass(T[n],fun_initdelta)    # do backward pass (6.3)-(6.5)
            self.setGradients(flagAccumulate)         # set gradients (6.6)-(6.7)
            flagAccumulate=1                          # accumulate gradients for n>=1
        if flagOptimize>0:                            # do learning step?
            self.lr_scheduler.update() # update learning rate scheduler
            self.optimizer.update()    # do weight update according to (6.8)-(6.9) or any other more refined optimization algorithm...
        return Y                                      # return outputs

    def predict(self,X):
        N=len(X)
        Y=np.zeros((N,self.K),self.par['stype']) # allocate memory for outputs
        for n in range(len(X)):
            self.forward_pass(X[n])
            Y[n,:]=self.layers[-1].z      # get and store output for x[n]
        return Y                          # return outputs

    def decide(self,Y,mode=None,thresh=None):   # make class decisions
        if mode is None: mode=self.par['modeClassify']
        if thresh is None: thresh=self.par['threshClassify']
        Y_hat=np.zeros(np.shape(Y),'byte')         # init decisions with zeros
        if (mode=='multi') or (mode=='single' and self.K==1):    # just do individual threshold operations?
            Y_hat[Y>=thresh]=1                     # set all elements 1 if >= thresh (multilabel)
        elif (mode=='max') or (mode=='single' and self.K>1):    # decide for maximal Y[n]
            Y_hat[np.arange(len(Y)),Y.argmax(1)]=1 # set max component to 1 in each row
        return Y_hat

    def getE(self,Y,T,eps=1e-8): # compute error/loss function value for given outputs Y and corresponding labels T
        assert self.par['loss'] in ['SSE','MSE','BCE','CCE'],"unknown loss function par['loss']="+str(par['loss'])
        assert Y.shape==T.shape,"Shapes of Y and T must be equal, but Y.shape"+str(Y.shape)+" and T.shape="+str(T.shape)
        E=None
        if self.par['loss'] in ['SSE','MSE']:
            D=Y-T
            E=0.5*np.sum(np.multiply(D,D))  # sum of squared error (6.20)
            if self.par['loss']=='MSE': E*=2.0/len(Y)
        elif self.par['loss']=='BCE':
            Y=np.maximum(Y,eps)
            Y=np.minimum(Y,1-eps)
            E=np.multiply(T,np.log(Y))+np.multiply(1-T,np.log(1-Y))
            E=np.sum(-E)
        elif self.par['loss']=='CCE':
            Y=np.maximum(Y,eps)
            Y=np.minimum(Y,1)
            E=np.multiply(T,np.log(Y))
            E=np.sum(-E)
        return E

    def getScore(self,Y_hat,T,score='error',mode=None):
        if mode is None: mode=self.par['modeClassify']
        assert score in ['error'],"currently only score=error is supported"
        assert mode in ['max','multi','single'],"currently only mode=max or mode=single or mode=multi is supported"
        assert Y_hat.shape==T.shape,"Shapes of Y_hat and T must be equal, but Y_hat.shape"+str(Y_hat.shape)+" and T.shape="+str(T.shape)
        if (mode=='single' and self.K==1) or mode=='multi':     # multilabels (multiple 1s in T[n] possible
            err=np.sum(np.abs(Y_hat-T))/float(T.size)
        elif (mode=='single' and self.K>1) or (mode=='max'):    # take max component in each row
            err = np.sum(Y_hat.argmax(1)-T.argmax(1)!=0)/float(T.shape[0])
        else:
            assert 0, "Invalid mode for the current setting"
        return err

    def eval_fit(self,X_train,T_train,X_val,T_val): # get key scores printed in fit(.)
        Y_train=self.predict(X_train)
        E_train=self.getE(Y_train,T_train)
        E_val,err_train,err_val='-','-','-'
        if not X_val is None:
            Y_val=self.predict(X_val)
            E_val=self.getE(Y_val,T_val)
        if self.par['flagClassify']>0:
            err_train=self.getScore(self.decide(Y_train),T_train)
            if not X_val is None: err_val=self.getScore(self.decide(Y_val),T_val)
        return E_train,err_train,E_val,err_val
    
    def fit(self,X_train,T_train,X_val=None,T_val=None,fun_initdelta=fun_initdelta_differror):
        assert len(X_train)==len(T_train),"X_train and T_train must have same length"
        if not X_val is None: assert not T_val is None, "T_val is None, although X_val is not!"  
        par=self.par
        batch_size=par['batch_size']
        N=len(X_train)
        addPerm=(batch_size-N%batch_size)%batch_size  # add this number of permutation samples to get full minibatches
        E,err,E_val,err_val=self.eval_fit(X_train,T_train,X_val,T_val)   # get initial performance measures
        print("Initial E=",E,"err=",err,"E_val=",E_val,"err_val=",err_val) 
        if not X_val is None: print("Initial validation error E_val=",self.getE(self.predict(X_val),T_val))
        self.epoch=0
        for self.epoch in range(par["maxEpochs"]):
            perm  = np.random.permutation(N)       # do a random permutation of data set
            if addPerm>0: np.concatenate(perm,perm[:addPerm])  # add perm such that we get full minibatches
            E=0.0      # initialize error
            for batch in range(len(perm)//batch_size):
                n1,n2=batch*batch_size,(batch+1)*batch_size
                X=X_train[perm[n1:n2]]
                T=T_train[perm[n1:n2]] 
                Y=self.fit_batch(X,T,fun_initdelta)
                E+=self.getE(Y,T)
            E,err,E_val,err_val=self.eval_fit(X_train,T_train,X_val,T_val)   # get performance measures
            print("After epoch=",self.epoch,"/",par["maxEpochs"],"E=",E,"err=",err,"E_val=",E_val,"err_val=",err_val) 
        
    def getGradient_numeric(self,X,T,eps=1e-5): # just for checking: compute gradient numerically 
        # (i) get parameters and allocate memory for numerical gradients
        weights=self.getWeights()   # get list of all weight arrays of the model
        gradients=[np.zeros(w.shape,self.par['stype']) for w in weights]  # get corresponding lists for gradients
        # (ii) accumulate partial derivatives computed numerically from differential quotients
        for j in range(len(weights)):   # loop over weights and gradients
            W,G=weights[j],gradients[j] # get references to j-th weight array and gradient array
            for i in range(W.size):     # loop over components of the weight array
                wi=W.flat[i]            # save weight value
                W.flat[i]=wi+eps        # increase i-th weight by eps
                Y=self.predict(X)       # evaluate network for increased weight
                E1=self.getE(Y,T)       # compute error for increased weight
                W.flat[i]=wi-eps        # decrease weight value
                Y=self.predict(X)       # evaluate network for decreased weight
                E2=self.getE(Y,T)       # compute error for decreased weight
                G.flat[i]=(E1-E2)/(2.0*eps) # difference quotient as approximation of partial derivative
                W.flat[i]=wi            # restore original weight
        return gradients
                
    def checkGradient(self,X,T,epsilon=1e-5,fun_initdelta=fun_initdelta_differror): # compare numeric gradient and gradient computed from backpropagation
        # (i) get numerical gradient
        gradients_num=self.getGradient_numeric(X,T,epsilon)   # get numeric gradient
        # (ii) compute gradient by backprop
        self.fit_batch(X,T,fun_initdelta,flagOptimize=0)      # do forward and backward pass for X,T
        gradients_bp=self.getGradients()                      # get list of backprop gradients
        # (iii) compute Euklidean norms of gradients and errors
        d=np.float128(0.0)            # init distance
        l_num=np.float128(0.0)        # init length of numeric gradients
        l_bp =np.float128(0.0)        # init length of backprop gradients
        for i in range(len(gradients_bp)):   # loop over all gradient arrays
            d_    =np.sum(np.multiply(gradients_bp[i]-gradients_num[i],gradients_bp[i]-gradients_num[i]))  # sum of squared errors
            l_num_=np.sum(np.multiply(gradients_num[i],gradients_num[i]))                                  # sum of squares
            l_bp_ =np.sum(np.multiply(gradients_bp[i],gradients_bp[i]))                                    # sum of squares
            #print("i=",i,"d_=",d_,"l_num_=",l_num_,"l_bp_=",l_bp_)
            #print("i=",i,"rel error=",2.0*d_/(l_num_+l_bp_))
            d+=d_
            l_num+=l_num_ 
            l_bp +=l_bp_ 
        d,l_num,l_bp=np.sqrt(d),np.sqrt(l_num),np.sqrt(l_bp)   # Euklidean distances and lengths
        abs_error=d                             # absolute error between numeric and backprop gradients
        rel_error=2.0*abs_error/(l_num+l_bp)    # corresponding relative error
        return rel_error

    def printState(self,flagData=0,flagWeights=0):
        print("Graph Neural Network:")
        print("Layers=",[str(l)+':'+self.layers[l].name for l in range(len(self.layers))])
        print("Connections=",[str(i)+':'+self.connections[i].name for i in range(len(self.connections))])
        print("Predecessors:")
        layer_idx={self.layers[i].name:i for i in range(len(self.layers))}   # l_idx[name] is index of layer with given name
        for l in range(len(self.layers)):
            str_l=str(l)+':'+self.layers[l].name+" <-- "
            sep=""
            for l_ in self.layers[l].pre_connections:
                str_l=str_l+sep+str(layer_idx[l_.layer_source.name])+":"+l_.layer_source.name
                sep=", "
            print("   ", str_l)
        print("Successors:")
        for l in range(len(self.layers)):
            str_l=str(l)+':'+self.layers[l].name+" --> "
            sep=""
            for l_ in self.layers[l].succ_connections:
                str_l=str_l+sep+str(layer_idx[l_.layer_target.name])+":"+l_.layer_target.name
                sep=", "
            print("   ", str_l)
        print("L=",self.L,"M=",self.M,"N_train=",self.N_train,"N_val=",self.N_val,"D=",self.D,"K=",self.K)
        print("par=",self.par) 
        if flagData>0:
            print("X_train=",self.X_train)
            print("T_train=",self.T_train)
            print("X_val=",self.X_val)
            print("T_val=",self.T_val)
        if flagWeights>0:
            pass



# *******************************************************
# __main___
# Module test
# *******************************************************

if __name__ == '__main__':
    # (i) create training data
    X1 = np.array([[-2,-1], [-2,2], [-1.5,1], [0,2], [2,1], [3,0], [4,-1], [4,2]])  # class 1 data
    N1,D1 = X1.shape
    X2 = np.array([[-1,-2],[-0.5,-1],[0,0.5],[0.5,-2],[1,0.5],[2,-1],[3,-2]])       # class 2 data
    N2,D2 = X2.shape
    T1,T2 = N1*[[1]],N2*[[0]]          # corresponding class labels (1 versus 0) 
    X = np.concatenate((X1,X2))        # entire data set
    T = np.concatenate((T1,T2))        # entire label set
    N,D = X.shape
    
    # (i) Define GraphNeuralNetwork
    par={}                             # dict of parameters
    par['lmbda']=1e-5                  # regularization coefficient
    par['eta0']=0.01                   # initial learning rate
    par['eta_fade']=0                  # fading factor for decreasing learning rate
    par['batch_size']=1                # number of input vectors per learning update 
    par['maxEpochs']=100               # number of learning epochs
    par['debug'] = 1                   # if >0 then debug mode: 1 = print Error, mean weight; 2=additionally check gradients; 3=additionally print weights
    par['wtype']='float64'             # floating point type for weights and biases
    par['stype']='float64'             # floating point type for state variables
    par['lr_scheduler']='simple_decay' # learning rate scheduler
    par['loss']='BCE'                  # loss/error function
    par['flagClassify']=1              # classification task 
    par['modeClassify']='single'       # classification mode: either 'single'-labels or 'multi'-labels
    par['threshClassify']=0.5          # decision threshold for classification
    par['optimizer']='MOMENTUM'             # optimizer algorithm (either SGD, ADAM, or MOMENTUM) 
    par['opt_beta']=0.9                # momentum decay factor
    par['opt_beta1']=par['opt_beta']   # Adam momentum decay
    par['opt_beta2']=0.999             # Adam variance decay
    par['opt_eps']=1e-8                # Adam epsilon
    gnn = GraphNeuralNetwork(par)
    gnn.addLayers([D,3,3,1],['x','z1','z2','y'],['linear','tanh','tanh','sigmoid'])   # create graph neural network with given layers
    gnn.addConnection(1,0,"W_1_0")
    gnn.addConnection(2,1,"W_2_1")
    gnn.addConnection(3,2,"W_3_2")
    gnn.addConnection(3,1,"W_3_1")     # short cut from layer 1 to layer 3 to go beyond simple sequential MLP
    gnn.compile()
    gnn.printState()
    rel_error=gnn.checkGradient(X,T)
    print("Check Gradients (Numerical vs. backprop): relative error=",rel_error)
    gnn.fit(X,T)

