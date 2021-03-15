import numpy as np
import pickle
import math 
import dataproc as dp


''' An implementation of an MLP with a single layer of hidden units. '''
class MLP:
    __slots__ = ('W1', 'b1', 'a1', 'z1', 'W2', 'b2', 'din', 'dout', 'hidden_units')

    def __init__(self, din, dout, hidden_units):
        ''' Initialize a new MLP with tanh activation and softmax outputs.
        
        Params:
        din -- the dimension of the input data
        dout -- the dimension of the desired output
        hidden_units -- the number of hidden units to use

        Weights are initialized to uniform random numbers in range [-1,1).
        Biases are initalized to zero.
        
       
        
        '''
        self.din = din
        self.dout = dout
        self.hidden_units = hidden_units 

        self.b1 = np.zeros((self.hidden_units, 1)) # bias for 1st layer
        self.b2 = np.zeros((self.dout, 1)) # bias for the second layer
        self.W1 = 2*(np.random.random((self.hidden_units, self.din)) - 0.5) ## weight vectors for the first layer
        self.W2 = 2*(np.random.random((self.dout, self.hidden_units)) - 0.5) ## weight vectors for the second layer


    def save(self,filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def load_mlp(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)
            
    def eval(self, xdata):
        ''' Evaluate the network on a set of N observations.

        xdata is a design matrix with dimensions (Din x N).
        This should return a matrix of outputs with dimension (Dout x N).
        See train_mlp.py for example usage.
        '''
        # input has to be din*N
        xdata = np.array(xdata).transpose() ## n * Din use each row as a single data point
        data_size =len(xdata) #number of datapoints N

        y= np.zeros((data_size,self.dout)) # N*dout
        
        
        self.z1= np.zeros((data_size,self.hidden_units))
        
        for index ,x in enumerate(xdata):
            z=[]
            # calculate the activiation of the hidden layer.
            for i in range (0,self.hidden_units):
                w= self.W1[i]
                z.append(math.tanh(np.dot(w,x)+float(self.b1[i])))
                # print(z)
            self.z1[index]=z

            a_out=[]

            #calculate the input for each output nodes
            for i in range(0,2):
                w=self.W2[i]
                
                a_out.append(np.dot(w,z)+self.b2[i])
            
            a_total=0
            
            num_output= len(a_out)
            #softmax
            for i in range(0,num_output) :
                a_total+= math.exp(a_out[i])

            
            temp =[]
            for i in range(0,num_output):
                temp.append(math.exp(a_out[i])/a_total)

            y[index]=temp
            
        return y.transpose() # return Dout *N


    def hidden_output (self,x):
        z1=[]
            # calculate the activiation of the hidden layer.
        for i in range (0,self.hidden_units):
            w= self.W1[i]
            z1.append(math.tanh(np.dot(w,x)+self.b1[i]))

        return z1

    def sgd_step(self, xdata, ydata, learn_rate):
        ''' Do one step of SGD on xdata/ydata with given learning rate. ''' 
        ## xdata: din *n  ydata :dout *n
     
       
       
       
        # W_upper= np.copy(self.W1)
        # W_lower= np.copy(self.W1)
        # W_upper[1][0]+=0.0001
        # W_lower[1][0]-=0.0001
      

        ##estimation for gradient of w11
        # est_gra=(self.error_cal(W_upper,self.W2,xdata,ydata)-self.error_cal(W_lower,self.W2,xdata,ydata))/0.0002 
        # print("est gra:{}".format(est_gra))

        gradient = self.grad(xdata,ydata)
        gw1= gradient[0]
      
      
        self.W1 -= (gw1*learn_rate)
       

        gb1= gradient[1]
        self.b1-= (gb1*learn_rate)

        gw2=gradient[2]
        
        
       
        self.W2-= (learn_rate*gw2)

        gb2=gradient[3]
        self.b2-=(learn_rate*gb2)
        
        
    def grad(self, xdata, ydata):
        ''' Return a tuple of the gradients of error wrt each parameter. 

        Result should be tuple of four matrices:
          (dE/dW1, dE/db1, dE/dW2, dE/db2)

        Note:  You should calculate this with backprop,
        but you might want to use finite differences for debugging.
        '''
        batch_size= len(xdata[0]) 

        for i in range(0,batch_size):
            x=xdata[:,i]
            t=ydata[:,i]
          
            # xdata is just a list, eclosing it by [] to make it transposable 2-d array
            predi_val= self.eval(np.array([x]).transpose()).transpose()
          

            t=t.transpose()

            #delta for the output units i.e. delta_k
            delta_k=predi_val[0]-t #dE/d a_k
            
           

            #delta for the hidden layer i.e. delta_j= dE/ d a_j
            delta_j=[]
            for j in range(0, self.hidden_units):
                sm = 0
                #k=0
                # for w_k in self.W2 :
                for k in range(0,len(self.W2)):
                    w_k=self.W2[k]
                    sm += w_k[j]*delta_k[k]
                    # k+=1
                w_j= self.W1[j]
              
                delta_j.append((1-(math.tanh(np.dot(w_j,x))**2))*sm)

            
            ew1=np.zeros(np.shape(self.W1))
            
            num_units = len(ew1)
            input_size= len(x)
            # calculation for dE/dW1
            for j in range(0,num_units):
                ew1_j=[] # gradient of weights to node j
                for i in range (0,input_size):
                   
                    ew1_j.append(x[i]*delta_j[j])

                ew1[j]= ew1_j

            # calculation for dE /d b1
            eb1=np.zeros(np.shape(self.b1))

            for j in range(0,len(eb1)):
                
                eb1[j]=delta_j[j]

            
            #calculation for dE/ dW2

            ew2 = np.zeros(np.shape(self.W2))
            num_op= len (ew2)
            z= self.z1[0]
            for k in range(0,num_op):
                ew2_k=[]
                for j in range(0, self.hidden_units):
                    ew2_k .append(delta_k[k]*(z[j]))
                ew2[k]=ew2_k

            #calculation for dE/db2
            eb2= np.zeros(np.shape(self.b2))
            
            for k in range (0,num_op):
                eb2[k]=delta_k[k]
                
        return (ew1,eb1,ew2,eb2)


    #Only for checking correctness of gradient.
    def error_cal (self ,W_1,W_2,xdata,ydata ):

        batch_size= len(xdata[0])
        sum_n=0
        for n in range(0,batch_size):
            x=xdata[:,n]
            t=ydata[:,n]
            
        
            z_j= []
            for j in range(0,self.hidden_units):
                w_j=W_1[j]
                
                z_j.append(math.tanh(np.dot(w_j,x))+float(self.b1[j]))

            a_to_k=[]
            for k in range(0,self.dout):
                w_k=W_2[k]
                
                a_to_k.append(np.dot(w_k,z_j)+float(self.b2[k]))
            
            a_total=0
            
            for a in a_to_k:
                
                a_total+= math.exp(a)
            
            y=[]
            for k in range (0,self.dout):
                y.append(math.exp(a_to_k[k])/a_total)

            sum_k=0
            for k in range(0,len(y)):
                sum_k-=t[k]*math.log(y[k])
            
            sum_n+=sum_k
        
        return sum_n

            


            


           


        

