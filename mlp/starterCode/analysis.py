import math
import numpy as np

def sum_squares(ydata, yhat):
    ''' Return complete sum of squared error over a dataset. PRML Eqn 5.11'''
    # PRML eqn 5.11
    ans = 0.0
    for n in range(ydata.shape[1]):
        ans += np.linalg.norm(ydata[:,n] - yhat[:,n])**2
    return ans/2

def mse(ydata, yhat):
    ''' Return mean squared error over a dataset. '''
    return sum_squares(ydata,yhat)/ydata.shape[1]

def cross_entropy(ydata, yhat):
    ''' Return cross entropy of a dataset.  See PRML eqn 5.24'''
    ans = 0
    for n in range(ydata.shape[1]):
        for k in range(ydata.shape[0]):
            if (ydata[k][n] * yhat[k][n] > 0):
                ans -= ydata[k][n] * math.log(yhat[k][n])
    return ans

def mce(ydata, yhat):
    ''' Return mean cross entropy over a dataset. '''
    return cross_entropy(ydata, yhat)/ydata.shape[1]

def accuracy(ydata, yhat):
    ''' Return accuracy over a dataset. ''' 
    correct = 0
    for n in range(ydata.shape[1]):
        if (np.argmax(ydata[:,n]) == np.argmax(yhat[:,n])):
            #print ("true:{}\npred:{}\n".format(np.argmax(ydata[:,n]),np.argmax(yhat[:,n])))
            correct += 1
    return correct / ydata.shape[1]

def f1(ydata,yhat):
    tp=0
    tn=0
    fp=0
    fn=0

    for n in range(ydata.shape[1]):
        pos1=np.argmax(ydata[:,n])
        pos2= np.argmax(yhat[:,n])

        if pos1==0 and pos2==0:
            tn+=1
            continue
        
        elif pos1 ==1 and pos2 ==1:
            tp+=1
            continue
        elif pos1 ==0 and pos2 ==1:
            fp+=1
            continue

        fn+=1

    if tp + fp ==0:
        print ("\nall elements in the batch are predicted to be in negative class")
        print ("ydata:\n{}\n".format(ydata))
        print ("yhat:\n{}\n".format(yhat))
        return -1
    p =tp/(tp+fp)

    if tp + fn == 0:
        print("all elements in the batch are not positive")
        return -1
    r=tp/(tp+fn)  
    
    if p+r ==0 :
        print("since {} both of them are 0, for convenience, we set f1 to 0")
        return 0
    return 2*p*r/(p+r)     
       




    
