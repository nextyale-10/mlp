import argparse
import numpy as np
import random
import analysis
import dataproc
import mlp
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--train_file', default=None, help='Path to the training data.')
    parser.add_argument('--dev_file', default=None, help='Path to the development data.')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train. (default 10)')
    parser.add_argument('--learn_rate', type=float, default=1e-1, help='The learning rate to use for SGD (default 1e-1).')
    parser.add_argument('--hidden_units', type=int, default=0, help='The number of hidden units to use. (default 0)')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use for SGD. (default 1)')
    args = parser.parse_args()


    # Load training and development data and convert labels to 1-hot representation.
    xtrain, ytrain = dataproc.load_data(args.train_file)
    ytrain = dataproc.to_one_hot(ytrain, int(1+np.max(ytrain[0,:])))
    if (args.dev_file is not None):
        xdev, ydev = dataproc.load_data(args.dev_file)
        ydev = dataproc.to_one_hot(ydev,int(1+np.max(ytrain[0,:])))

    # Record dimensions and size of dataset.
    N = xtrain.shape[1] # the number of data points
    # print("xtrain:\n{}\n".format(xtrain))

    din = xtrain.shape[0]
    dout = ytrain.shape[0]

    batch_size = args.batch_size
    if (batch_size == 0):
        batch_size = N
    
    # Create an MLP object for training.
    nn = mlp.MLP(din, dout, args.hidden_units)
    #print("w1:\n{}\n\nw2:\n{}\n".format(nn.W1,nn.W2))

    # Evaluate MLP after initialization; yhat is matrix of dim (Dout x N).
    yhat = nn.eval(xtrain)
    #print (yhat)

    best_train = (analysis.mse(ytrain, yhat), 
                  analysis.mce(ytrain, yhat),
                  analysis.accuracy(ytrain, yhat)*100)
    print('Initial conditions~~~~~~~~~~~~~')
    print('mse(train):  %f'%(best_train[0]))
    print('mce(train):  %f'%(best_train[1]))
    print('acc(train):  %f'%(best_train[2]))
    print("F1(train):  {}".format(analysis.f1(ytrain,yhat)))
    print('')
    
    if (args.dev_file is not None):
        best_dev = (analysis.mse(ydev, yhat), 
                      analysis.mce(ydev, yhat),
                      analysis.accuracy(ydev, yhat)*100)
        print('mse(dev):  %f'%(best_dev[0]))
        print('mce(dev):  %f'%(best_dev[1]))
        print('acc(dev):  %f'%(best_dev[2]))
        print("F1(dev):  {}".format(analysis.f1(ydev,yhat)))
    epo=[]
    acc_train=[]
    acc_dev=[]
    f1_train=[]
    f1_dev=[]
    mce_train=[]
    mce_dev=[]
    name= args.train_file.split("/")
    f_name= name[-1].split(".")[0]
    f_name+= "_{}_{}_{}".format(args.epochs,args.learn_rate,args.hidden_units)
    counter= 0
    for epoch in range(args.epochs):
        # stuck = False
        # threshold=0.0015
        # change_rate=0
        for batch in range(int(N/batch_size)):
            ids = random.choices(list(range(N)), k=batch_size)
            # print("ids:{}".format(ids))
            xbatch = np.array([xtrain[:,n] for n in ids]).transpose()
            # print("________________________\n{}".format(xbatch))
            ybatch = np.array([ytrain[:,n] for n in ids]).transpose()
            # print("\n{}\n\n".format(ybatch))
            nn.sgd_step(xbatch, ybatch, args.learn_rate)
            #print("\n\n___________________________\nw1:\n{}\n\nw2:\n{}\n".format(nn.W1,nn.W2))

        
        yhat = nn.eval(xtrain)
        train_ss = analysis.mse(ytrain, yhat)
        train_ce = analysis.mce(ytrain, yhat)
        train_acc = analysis.accuracy(ytrain, yhat)*100
        best_train = (min(best_train[0], train_ss), min(best_train[1], train_ce), max(best_train[2], train_acc))
        train_f1=analysis.f1(ytrain,yhat)
        epo.append(epoch)
        acc_train.append(train_acc)
        f1_train.append(train_f1)
        mce_train.append(train_ce)
        
        # if (counter == 10 ):
        #     args.learn_rate= args.learn_rate/10
        #     print("learning rate is now {}".format(args.learn_rate))
        #     counter=0
        # if epoch >10:
        #     change_rate= (abs(mce_train[epoch]-mce_train[epoch-1])/mce_train[epoch]) 
        # if change_rate<threshold:
        #     counter +=1

        fig= plt.figure("Accuracy for {}({} hidden units with batch size {}: LR {})".format(f_name,args.hidden_units,args.batch_size,args.learn_rate))
        print('After %d epochs ~~~~~~~~~~~~~'%(epoch+1))
        print('mse(train):  %f  (best= %f)'%(train_ss, best_train[0]))
        print('mce(train):  %f  (best= %f)'%(train_ce, best_train[1]))
        print('acc(train):  %f  (best= %f)'%(train_acc, best_train[2]))
        print("F1(train):  {}".format(train_f1))

        if (args.dev_file is not None):
            yhat = nn.eval(xdev)
            dev_ss = analysis.mse(ydev, yhat)
            dev_ce = analysis.mce(ydev, yhat)
            dev_acc = analysis.accuracy(ydev, yhat)*100
            dev_f1=analysis.f1(ydev,yhat)
            acc_dev.append(dev_acc)
            f1_dev.append(dev_f1)
            mce_dev.append(dev_ce)
            best_dev = (min(best_dev[0], dev_ss), min(best_dev[1], dev_ce), max(best_dev[2], dev_acc))
            print('mse(dev):  %f  (best= %f)'%(dev_ss, best_dev[0]))
            print('mce(dev):  %f  (best= %f)'%(dev_ce, best_dev[1]))
            print('acc(dev):  %f  (best= %f)'%(dev_acc, best_dev[2]))
            print("F1(dev):  {}".format(dev_f1))
            

        print('')
        
       
        nn.save(f_name)

    plt.plot(epo,acc_train)
    if (args.dev_file is not None):
        
        plt.plot(epo,acc_dev)
    plt.legend(["train_file","dev_file"])
    plt.xlabel("epochs")
    plt.ylabel("accuracy")

    plt.figure("F1 for {}({} hidden units with batch size {}: LR {})".format(f_name,args.hidden_units,args.batch_size,args.learn_rate))
    plt.plot(epo,f1_train)
    if (args.dev_file is not None):
        
        plt.plot(epo,f1_dev)
    plt.legend(["train_file","dev_file"])
    plt.xlabel("epochs")
    plt.ylabel("F1")


    plt . figure("mce for {}({} hidden units with batch size {}: LR {})".format(f_name,args.hidden_units,args.batch_size,args.learn_rate))
    plt.plot(epo,mce_train)
    if (args.dev_file is not None):
        plt.plot(epo,mce_dev)
    plt.legend(["train_file","dev_file"])
    plt.xlabel("epochs")
    plt.ylabel("mce")

    plt.show()

if __name__ == '__main__':
    main()
