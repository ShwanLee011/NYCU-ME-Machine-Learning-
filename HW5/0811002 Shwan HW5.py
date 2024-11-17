#Hw5 Handout
#target: avg.validation score-->the higher, the better. Test_score>=0.940
import sklearn
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
############################################################
def readHw5Cancer(inFileName):
    # init
    recArr = []
    clsArr = []
    
    # open input text data file, format is given
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
        
        strs31 = data1.split(',') # array of 31 str

        # convert to real
        for ix in range(30):
            value.append( eval(strs31[ix]) )
        # end for
        
        target = eval(strs31[30])

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

# end function

#Main starts
X,y=readHw5Cancer("d:\\temp\\breast_cancer_scikit_Xy.csv")

#print(X_5fold.shape)
#print(X_test.shape)
totalnum=569
nfold = 5
mxi = 100000

#LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 94, random_state = 0)
cvalue=[0.1,1,10,100,1000]
lavgscores = []
print("(1)Logistic Regression:")
for C in cvalue:
    logreg = LogisticRegression(C=C,max_iter=mxi).fit(X_train,y_train)
    train_score = logreg.score(X_train,y_train)
    scores = cross_val_score(logreg,X_train,y_train,cv=nfold,scoring='accuracy')
    test_score = logreg.score(X_test,y_test)
    lavg = 0.5*(test_score+scores.mean())
    print("C="+str(C))
    print('5-fold cross validation scores'+str(scores))
    print("5-fold cross validation average score: {:.3f}".format(scores.mean()))
    print("train score={:.3f}, test score= {:.3f}".format(train_score,test_score))
    lavgscores.append(lavg)
max_lscore = 0.
max_i = 0
for i, num in enumerate(lavgscores):
    if (max_lscore == 0 or num > max_lscore):
        max_lscore = num
        max_i = i

print("(1)Maximum average test score is {:.3f}, when C value equal to {:.2f}. ".format(max_lscore,cvalue[max_i]))

#RandomForest
print("(2)Random Forest: ")
favgscores=[]
estinum = list(range(100,210,10))
for ix in estinum:
    forest = RandomForestClassifier(n_estimators=ix ,criterion="gini", max_depth=None, bootstrap=True, random_state=None)
    forest.fit(X_train, y_train)
    ftrains = forest.score(X_train,y_train)
    fscores = cross_val_score(forest,X_train,y_train,cv=nfold,scoring='accuracy')
    ftests = forest.score(X_test,y_test)
    favg = 0.5*(ftests+fscores.mean())
    favgscores.append(favg)
    print("number of estimators ={:d} ".format(ix))
    print("5-fold cross validation scores:"+str(fscores))
    print("5-fold cross validation average score: {:.3f}".format(fscores.mean()))
    print("score Train/Test: {:.3f}/{:.3f}".format(ftrains,ftests))
    #print(ftrains)
max_fscore = 0.
max_j = 0
for j, num in enumerate(favgscores):
    if (max_fscore == 0 or num > max_fscore):
        max_fscore = num
        max_j = j
print("(2)Maximum test score is {:.3f}, while number of estimators is {:d}. ".format(max_fscore,estinum[max_j]))

#GradientBoosting
learnrate = [0.01,0.1,1,10,100,1000]
gavgscores = []
print("(3)GradientBoosting")
for x in learnrate:
    gbrt = GradientBoostingClassifier(learning_rate=x,n_estimators=100,random_state=0)
    gbrt.fit(X_train,y_train)
    g_test = gbrt.score(X_test,y_test)
    g_train = gbrt.score(X_train,y_train)
    gscores = cross_val_score(gbrt,X_train,y_train,cv=nfold,scoring='accuracy')
    gavg = 0.5*(g_test+gscores.mean())
    gavgscores.append(gavg)
    print("Learning rate is {:.3f}".format(x))
    print("5-fold cross validation scores: "+str(gscores))
    print("5-fold cross validation average score: {:.3f}".format(gscores.mean()))
    print("train/test scores:{:.3f}/{:.3f} ".format(g_train,g_test))
max_gscore = 0.
max_j = 0
for k, num in enumerate(gavgscores):
    if (max_gscore == 0 or num > max_gscore):
        max_gscore = num
        max_k = k
print("(3)Maximum test score is {:.3f}, while learning rate is {:.3f}. ".format(max_gscore, learnrate[max_k]))
