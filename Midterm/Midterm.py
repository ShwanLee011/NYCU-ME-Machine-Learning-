#Midterm resubmit
import mglearn
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
def ReadData(file_name):
    x=[]
    y1=[]

    f_in=open(file_name, 'r')
    for i in range(73):
        s=f_in.readline()
        #print(type(s))
        if (i<=60):
            if(i!=0):
                #s=f_in.readline()
                s=s.replace('[','')
                s=s.replace(']','')
                #s=s.strip(',')
                s=s.split()
                s=str(s)
                x.append(eval(s))
        elif(i>61):
            s=s.replace('[','')
            s=s.replace(']','')
            s=s.strip(',')
            s=s.split()
            #print(s)
            s = str(s)
            y1.append(eval(s))
            y=np.zeros([len(y1),len(max(y1,key = lambda x:len(x)))])
            for i,j in enumerate(y1):
                y[i][0:len(j)] = j
    print(y1)
    f_in.close()
    x=np.array(x)
    x.reshape(-1,60)
    newy=[]
    y=np.array(y)
    for i in range (11):
        for j in range(6):
            if(y[i][j]!=0.):
                newy.append(y[i][j])
    newy = np.array(newy)
    newy.reshape(10,-1)
    return x,newy
#def 
x,y=ReadData('D:\\temp\\wave60_dataset.txt')
x = np.array(x, dtype=float)
X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=10,random_state=0)
for i in range(1,11,2):
    knn=KNeighborsRegressor(n_neighbors=i,weights="uniform")
    knn.fit(X_train,y_train)
    ktrain = knn.score(X_train,y_train)
    ktest = knn.score(X_test,y_test)
    print("uniform, KNN={:d}, X_test/X_train score = {:.2f}/{:.2f}".format(i,ktest,ktrain))
for i in range(1,11,2):
    knn=KNeighborsRegressor(n_neighbors=i,weights="distance")
    knn.fit(X_train,y_train)
    ktrain = knn.score(X_train,y_train)
    ktest = knn.score(X_test,y_test)
    print("distance, KNN={:d}, X_test/X_train score = {:.2f}/{:.2f}".format(i,ktest,ktrain))    
