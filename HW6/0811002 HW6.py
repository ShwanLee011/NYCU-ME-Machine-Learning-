import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
def readHw6File(inFileName):
    # init
    recArr = []
    clsArr = []
    
    # open input text data file, format is givens
    inFile = open(inFileName, 'r')
    s = inFile.readline() # skip
    
    row = 0
    while True:
        s = inFile.readline()
        data1 = s.strip() # remove leading and ending blanks
        if (len(data1) <= 0):
            break
        
        # since we use append, value must be created in the loop
        value = []
        
        strs3 = data1.split(',') # array of 31 str

        # convert to real
        for ix in range(3):
            value.append( eval(strs3[ix]) )
        # end for
        
        target = eval(strs3[3])

        recArr.append(value) ;  # add 1 record at end of array
        clsArr.append(target) ; # add 1 record at end of array
       
        row = row+1 # total read counter
    # end while
    
    # close input file
    inFile.close()

    # convert list to Numpy array
    npXY = np.array(recArr)
    npC  = np.array(clsArr)

    # pass out as Numpy array
    return npXY, npC
#Main start
#Readin data
X,y=readHw6File("d:\\temp\\hw6_haberman.csv")
#1.MLPclassifier part
#set different activation functions to choose
actfunc={'identity', 'logistic', 'tanh', 'relu'}
#set different solver funcitons to choose
solverfunc={'lbfgs', 'sgd', 'adam'}
#set different hidden layer sizes to choose
hidsize={10,20,30,100,200,300}
#use 3-layer for loop to change parameters every time
for i in actfunc:
    for j in solverfunc:
        for k in hidsize:
            
            mlp = MLPClassifier(activation=i,hidden_layer_sizes=(k,),max_iter=200000
                         ,alpha=0.01,solver=j,verbose=10,random_state=1,learning_rate_init=0.001)
            mlp.fit(X,y)
#only print score and parameters when training score larger than 0.90
            if mlp.score(X,y)>=0.90:
            
                print('Hidden layer size is '+str(k)+',activation function is '+str(i)+', and solver func is '+str(j)+'.')
                print('Training score is {:.3f}'.format(mlp.score(X,y)))
#2.SVC part
#set different C values to choose
Cvalue=[0.1,1.0,3.0,10.0]
#set different kernel functions to choose('precomputer' need to input square matrix)
kernelfunction={'linear', 'poly', 'rbf', 'sigmoid'}
#set different gamma value to choose
rvalue=[0.001,0.01,0.1,1.0]
#use 3-layers for loop to train model
for c in Cvalue:
    for func in kernelfunction:
        for r in rvalue:
            svc=SVC(C=c, kernel=func, degree=3, gamma=r,max_iter=-1, decision_function_shape='ovo',random_state=None)
            svc.fit(X,y)
#only print score and parameters when training score larger than 0.90
            if svc.score(X,y)>=0.90:
                print('SVC with C value '+str(c)+', kernel function '+str(func)+', and gamma value '+str(r)+'.')
                print('The score is: {:.3f}'.format(svc.score(X,y)))
#set gamma to be auto here
for c in Cvalue:
    for func in kernelfunction:
        svc=SVC(C=c, kernel=func, degree=3, gamma='auto',max_iter=-1, decision_function_shape='ovo',random_state=None)
        svc.fit(X,y)
#only print score and parameters when training score larger than 0.90
        if svc.score(X,y)>=0.90:
            print('SVC with C value '+str(c)+', kernel function '+str(func)+', and auto gamma value.')
            print('The score is: {:.3f}'.format(svc.score(X,y)))


