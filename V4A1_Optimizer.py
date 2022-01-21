# UA6_3a_Optimizer.py 
import numpy as np

def optimize_SGD(w,eta,nablaE): # compute update step for SGD
    return w-eta*nablaE(w)      # SGD update,see (6.105)  

def optimize_momentum(w,m,eta,beta,nablaE): # compute update step for momentum 
    #m=???                      # update momentum, see (6.106)
    #w=???                      # update weights, see (6.107)
    return w,m

def optimize_Nesterov(w,m,eta,beta,nablaE): # compute update step for Nesterov 
    #m=???                      # update momentum, see (6.108)
    #w=???                      # update weights, see (6.109)
    return w,m

def optimize_Adagrad(w,v,eta,eps,nablaE): # compute update step for Adagrad
    #g=???                      # get gradient, see (6.110)
    #v=???                      # add squared gradients, see (6.111)
    #w=???                      # weight update, see (6.112) and (6.113)
    return w,v

def optimize_RMSprop(w,v,eta,gamma,eps,nablaE): # compute update step for RMSprop
    #g=???                      # get gradient, see (6.110)
    #v=???                      # moving average of variance, see (6.115)
    #w=???                      # weight update, see (6.116)
    return w,v

def optimize_NewtonRaphson(w,nablaE,HE): # compute update step for 2nd order Newton-Raphson algorithm (HE is Hessian of E)
    #w=???                      # Newton-update, see (6.118)
    return w

def optimize_Adadelta(w,v,v_dw,gamma,eps,nablaE): # compute update step for Adadelta
    #g=???                      # get gradient, see (6.110)
    #v=???                      # moving average of gradient variance, see (6.115)
    #dw=???                     # weight update, see (6.123)
    #v_dw=???                   # moving average of weight delta variance, see (6.115) with footnote 60 on page 49 of SB6
    #w=???                      # weight update
    return w,v,v_dw

def optimize_ADAM(w,m,v,tau,eta,beta1,beta2,eps,nablaE): # compute update step for ADAM
    #g=                         # get gradient for weights w
    #m=                         # update first moment, see (6.124)
    #v=                         # update second moment, see (6.125)
    #m_hat=                     # corrected first moment (6.126)
    #v_hat=                     # corrected second moment (6.126)
    #print("m=",m,"v=",v,"m_hat=",m_hat,"v_hat=",v_hat)
    #w=                         # weight update (6.127)
    return w,m,v

def getOptimizationTrajectory(alg,w0,T,par,nablaE,HE=None,E=None,debug=0):
    assert alg in ['SGD','MOMENTUM','NESTEROV','ADAGRAD','RMSprop','NEWTON','ADADELTA','ADAM'],"Unknown optimizer "+str(alg)
    w=np.array(w0,'float')         # initial weights
    m=np.zeros(w.shape,'float')    # initialize m with zeros 
    v=np.zeros(w.shape,'float')    # initialize v with zeros 
    v_dw=np.zeros(w.shape,'float') # initialize v_dw with zeros
    w_list=[w]
    tau=0
    while tau<T:
        tau=tau+1
        if alg=='SGD':
            w=optimize_SGD(w,par['eta'],nablaE)
        elif alg=='MOMENTUM':
            w,m=optimize_momentum(w,m,par['eta'],par['beta'],nablaE)
        elif alg=='NESTEROV':
            w,m=optimize_Nesterov(w,m,par['eta'],par['beta'],nablaE)
        elif alg=='ADAGRAD':
            w,v=optimize_Adagrad(w,v,par['eta'],par['eps'],nablaE)
        elif alg=='RMSprop':
            w,v=optimize_RMSprop(w,v,par['eta'],par['gamma'],par['eps'],nablaE)
        elif alg=='NEWTON':
            w=optimize_NewtonRaphson(w,nablaE,HE)
        elif alg=='ADADELTA':
            if tau==1:
                dw=optimize_SGD(w,par['eta'],nablaE)-w  # get dw
                v_dw=np.multiply(dw,dw)                 # init v_dw
            w,v,v_dw=optimize_Adadelta(w,v,v_dw,par['gamma'],par['eps'],nablaE)
        elif alg=='ADAM':
            w,m,v=optimize_ADAM(w,m,v,tau,par['eta'],par['beta1'],par['beta2'],par['eps'],nablaE)
        if(debug>0):
            if not E is None: Ew=E(w)
            else: Ew=None
            print("\ntau=",tau,"w=",w,"m=",m,"v=",v,"v_dw=",v_dw,"nablaE=",nablaE(w),"E=",Ew)
        w_list+=[w]
    return w_list

def E_quadratic(w,A,b): # general quadratic E(w)=0.5*wT*A*w-bT*w
    return 0.5*np.dot(w.T,np.dot(A,w))-np.dot(b.T,w) 
    
def nablaE_quadratic(w,A,b):  # compute gradient for quadratic E(w)=0.5*wT*A*w-bT*w
    return 0.5*np.dot(w.T,A+A.T)-b.T  # gradient = 2*wT*(A+AT)^(-1)-bT
   
# ******************** main program *******************
if __name__ == '__main__':
    # (i) define quadratic function and its gradient
    A=np.array([[2,1],[1,3]],'float')   # matrix A
    b=np.array([1,-2],'float')          # vector b
    E=lambda w : E_quadratic(w,A,b)     # loss function
    nablaE=lambda w : nablaE_quadratic(w,A,b) # gradient of loss
    HE=lambda w: 0.5*(A+A.T)     # Hessian of loss (only for Newton algorithm)
    w_min=2.0*np.dot(np.linalg.inv(A+A.T),b)  # actual minimum location
    print("For A=",A,"the eigenvalues of A+A.T are",np.linalg.eig(A+A.T)[0])
    print("If both are positive, then a unique minimum exists at w_min=",w_min,"\n")
    
    # (ii) define parameters
    par={}      # init parameter dict
    par['eta'  ]=0.01          # learning rate
    par['beta1']=0.9           # decay rate for ADAM momentum
    par['beta2']=0.999         # decay rate for ADAM variance
    par['eps'  ]=1e-8          # numeric stability parameter
    par['beta' ]=par['beta1']  # decay rate for momentum
    par['gamma']=par['beta1']  # decay rate for Adadelta
    alg='SGD'   # choose between 'SGD', 'MOMENTUM', 'NESTEROV', 'ADAGRAD', 'RMSprop', 'NEWTON', 'ADADELTA', 'ADAM'
    w0=np.array([2,1],'float') # initial weights
    T=10        # number of learning steps (use 3 to verify Uebungsaufgabe 6.11.c; use 500 to observe convergence) 
    debug=1     # if >0 then print detailed information
    
    # (iii) compute optimization trajectory by doing T learning steps
    print("Do T=",T," learning steps using alg=",alg)
    print("with parameters eta=",par['eta'],"beta1=",par['beta1'],"beta2=",par['beta2'],"eps=",par['eps'],"beta=",par['beta'])
    print("\nw0=",w0,"nablaE=",nablaE(w0),"E=",E(w0))
    w_list=getOptimizationTrajectory(alg,w0,T,par,nablaE,HE,E,debug)
    print("\nafter learning: w=",w_list[-1],"nablaE=",nablaE(w_list[-1]),"E=",E(w_list[-1]))
    print("w_min=",w_min,"E_min=",E(w_min),"nablaE_min=",nablaE(w_min)) # actual minimum 
    
    # (iv) plot error function and weight trajectory
    import matplotlib.pyplot as plt
    w1_range,w2_range=[0,3,0.2],[-2,2,0.2]
    W1 = np.arange(w1_range[0], w1_range[1], w1_range[2])
    W2 = np.arange(w2_range[0], w2_range[1], w2_range[2]) 
    W1, W2 = np.meshgrid(W1, W2)
    Egrid = np.zeros(W1.shape,'float')
    w=np.zeros(2,'float')
    for i in range(W1.shape[0]):
        for j in range(W2.shape[1]):
            w[0],w[1]=W1[i,j],W2[i,j]
            Egrid[i,j]=E_quadratic(w,A,b)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([w_list[0][0]], [w_list[0][1]], c='r', marker='o', s=100)
    ax.scatter([w_min[0]], [w_min[1]], c='g', marker='*', s=200)
    ax.plot([w[0] for w in w_list],[w[1] for w in w_list],'rx:')
    CS=ax.contour(W1,W2,Egrid)
    ax.grid()
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_title(alg+" eta="+str(par['eta'])+" beta1="+str(par['beta1'])+" beta2="+str(par['beta2'])+" eps="+str(par['eps'])+" beta="+str(par['beta']))
    plt.show()
