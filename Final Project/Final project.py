import numpy as np
import sklearn
#First model I used
from sklearn.neighbors import KNeighborsRegressor
#Second model I used
from sklearn.neural_network import MLPRegressor
#I need to do cross validation
from sklearn.model_selection import cross_val_score
#Third model I used
from sklearn.svm import SVR

def readMLB(inFileName):
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
        
        strs45 = data1.split(',') # array of 31 str

        # convert to real
        for ix in range(44):
            value.append(eval(strs45[ix]))
        # end for
        
        target = int(strs45[43])

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
#Step1:read all data I need for training and testing. Total 6 CSV files are included in RAR file.
#NL data
NL_traindata,NL_trainwins=readMLB('d:\\大學課業\\大四\\ML\\Final Project\\NL Training Data.csv')
NL_testdata,NL_realwins=readMLB('d:\\大學課業\\大四\\ML\\Final Project\\NL Test Data.csv')
#AL data
AL_traindata, AL_trainwins=readMLB('d:\\大學課業\\大四\\ML\\Final Project\\AL Training Data.csv')
AL_testdata,AL_realwins=readMLB('d:\\大學課業\\大四\\ML\\Final Project\\AL Test Data.csv')
#Full data
Full_traindata, Full_trainwins=readMLB('d:\\大學課業\\大四\\ML\\Final Project\\MLB full training.csv')
Full_testdata,Full_realwins=readMLB('d:\\大學課業\\大四\\ML\\Final Project\\MLB fulltest.csv')

#Step2: start trying different models
#(1)n_neighbor regressor
#parameters changed: n_neighbors
#Best case: 7 neighbors
num=np.arange(1,9,2)
#create an nparray include number of neighbors I want to try
#i.train NL data
for i in num:    
    knnNL=KNeighborsRegressor(n_neighbors=i,weights="distance")
    knnNL.fit(NL_traindata,NL_trainwins)
    knnNLs=cross_val_score(knnNL,NL_traindata,NL_trainwins,cv=5)
    print('n_neighbors='+str(i))
    print(knnNL.predict(NL_testdata).astype(int))
    print('5-fold validaiton scores are: '+str(knnNLs))
    print('5-fold average score is: {:.3f}'.format(knnNLs.mean()))
    print('\n')

#ii.train AL data
for j in num:
    knnAL=KNeighborsRegressor(n_neighbors=j,weights="distance")
    knnAL.fit(AL_traindata, AL_trainwins)
    knnALs=cross_val_score(knnAL,NL_traindata,NL_trainwins,cv=5)
    print('n_neighbors='+str(j))
    print('5-fold validaiton scores are: '+str(knnALs))
    print(knnAL.predict(AL_testdata).astype(int))
    print('5-fold average score is: {:.3f}'.format(knnALs.mean()))
    print('\n')
#iii.train full dataset
for k in num:
    knnall=KNeighborsRegressor(n_neighbors=j,weights="distance")
    knnall.fit(Full_traindata, Full_trainwins)
    knnalls=cross_val_score(knnAL,NL_traindata,NL_trainwins,cv=5)
    print('n_neighbors='+str(j))
    print('5-fold validaiton scores are: '+str(knnalls))
    print(knnall.predict(Full_testdata).astype(int))
    print('5-fold average score is: {:.3f}'.format(knnalls.mean()))
    print('\n')
#end of n_neighbor regressor


#(2)MLP Regressor
#Parameter I changed: activation function
#hidden_layer size=6 is the most stable case after trying many times
actfunc={'identity', 'logistic', 'tanh', 'relu'}

#Best Case:  solver func:'lbfgs', activation:'identity'
for fun in actfunc:
        mlp = MLPRegressor(hidden_layer_sizes=(6,), activation=fun,solver='lbfgs', 
                                                        alpha=0.001,batch_size='auto', learning_rate="constant",learning_rate_init=0.0001,power_t=0.5, max_iter=500000,tol=1e-5)

        #NL data
        mlp.fit(NL_traindata,NL_trainwins)
        print('NL predict with '+str(fun)+' and '+str('lbfgs')+': '+str(mlp.predict(NL_testdata).astype(int)))
        mlpscores=cross_val_score(mlp,NL_traindata,NL_trainwins,cv=5)
        print(mlpscores)
        print('{:.5f}'.format(mlpscores.mean()))
        print('\n')
        #AL data
        mlp.fit(AL_traindata, AL_trainwins)
        print('AL predict with '+str(fun)+' and '+str('lbfgs')+': '+str(mlp.predict(AL_testdata).astype(int)))
        mlpscores=cross_val_score(mlp,AL_traindata,AL_trainwins,cv=5)
        print(mlpscores)
        print('{:.5f}'.format(mlpscores.mean()))
        print('\n')
        #Full data
        mlp.fit(Full_traindata, Full_trainwins)
        print('Full predict with '+str(fun)+': '+str(mlp.predict(Full_testdata).astype(int)))
        mlpscores=cross_val_score(mlp,Full_traindata,Full_trainwins,cv=5)
        print(mlpscores)
        print('{:.5f}'.format(mlpscores.mean()))
        print('\n')
#start SVR part
#parameter I changed:kernel function
#C=0.1 is the best. If C=1.0, it tends to be overfitting.
kernelfunction={'linear','rbf','sigmoid'}
for func in kernelfunction:
    svr=SVR(C=.01, kernel=func, degree=4, gamma='auto',tol=0.001, max_iter=-1)
    #NL data
    svr.fit(NL_traindata,NL_trainwins)
    print('NL Predict Wins with kernelfunction'+str(func)+' are: '+str(svr.predict(NL_testdata).astype(int)))
    svrs=cross_val_score(svr,NL_traindata,NL_trainwins,cv=5)
    print('5-fold cross validaiton scores are: '+str(svrs))
    print(svrs.mean())
    #AL data
    svr.fit(AL_traindata, AL_trainwins)
    print('AL Predict Wins with kernelfunction'+str(func)+' are: '+str(svr.predict(AL_testdata).astype(int)))
    svrs=cross_val_score(svr,AL_traindata, AL_trainwins,cv=5)
    print('5-fold cross validaiton scores are: '+str(svrs))
    print(svrs.mean())
    #Full data
    svr.fit(Full_traindata, Full_trainwins)
    print('All Predict Wins with kernelfunction'+str(func)+'are: '+str(svr.predict(Full_testdata).astype(int)))
    svrs=cross_val_score(svr,Full_traindata, Full_trainwins,cv=5)
    print('5-fold cross validaiton scores are: '+str(svrs))
    print(svrs.mean())
#Part3: print real wins in 2022 to compare the predict result
print('Real NL wins are:'+str(NL_realwins))
print('Real AL wins are:'+str(AL_realwins))
print('Real wins are:'+str(Full_realwins))


