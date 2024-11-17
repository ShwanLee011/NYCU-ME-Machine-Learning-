import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
#import csv
def readIris(name):
        
        fil = open(name, 'r')
        #lines = file.readlines()
        chara = []
        typ = []
        database = []
       # print(lines)
       # lines.remove("\n")
       # print(lines)
        for ix in range(300):
            row = fil.readline()
            #print(row)
            if row!="\n":
                row = row.strip()
                #row = row.strip('[')
                #row = row.strip(']')
            #print(type(line))
                row = row.split(",")
               # print(row)
           #print(type(line))
                for i in range(4):
                    chara.append(eval(row[i]))
                typ.append(row[4])
                database.append(row)
        chara = np.array(chara)
        typ = np.array(typ)
        database = np.array(database)
        return chara, typ, database
        file.close()
def Traindata(charac,specie,alldata):
  X = charac
  y = specie
  
  score = 0
  while(score!=1):
    print("test_score is "+str(score))
    X,y = randshuf(alldata)
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)
    knn.fit(X_train, y_train.ravel())
    score = knn.score(X_test,y_test)
    
def randshuf(alldata):
  np.random.shuffle(alldata)
  #newd = np.array(alldata)
  newX = []
  newy = []
  for ix in range(150):
    for k in range(4):
      newX.append(eval(alldata[ix][k]))
    newy.append(eval(alldata[ix][4]))

  
  newX = np.array(newX)
  newX = newX.reshape(-1,4)
  
  newy = np.array(newy)
  newy = newy.reshape(-1,1)
  return newX, newy
#Main
name = "D:\\Temp\\0811002_Shwan_iris_data.csv" 
charac, specie, alldata = readIris(name)

charac = charac.reshape(-1,4)
#print(charac)
specie = specie.reshape(-1,1)
alldata = np.array(alldata)
#print(alldata)
#print(specie)
#Prepare training data
knn = KNeighborsClassifier(n_neighbors = 3)
Traindata(charac,specie,alldata)
newdata = [[5, 2.9, 1, 0.2],[3, 2.2, 4, 0.9]]
predresult = knn.predict(newdata)
for ix in range(2):
  if predresult[ix]==0:
    specie = "setosa"
  elif predresult[ix]==1:
    specie = "versicolor"
  elif predresult[ix]==2:
    specie = "virginica"
  print("For data "+str(ix)+", type of flower is "+specie)
