#HW2 程式碼
database = "d:\大學課業\大四\ML\HW2\iris_dataset.txt"
irisdata = open(database,'r')
num = 150

import matplotlib.pyplot as plt
from numpy import random
import numpy as np
import csv
def number (species) :
    seto=0
    veri=0
    virg=0
    for ix in species:
        if ix == '0':
            seto=seto+1
        elif ix == '1':
            veri=veri+1
        elif ix == '2':
            virg=virg+1
    print("No5.Number of Setosa = "+str(seto)+", Verscolor ="+str(veri)+", and Virgina ="+str(virg)+".")
#最後五行0,1,2，找出三種的數量
def findmed(spec1,spec2,spec3,spec4):
    state=[]
    #species1=np.array(spec1)
    #species2=np.array(spec2)
    state.append(np.median(spec1))
    state.append(np.median(spec2))
    state.append(np.median(spec3))
    state.append(np.median(spec4))
    for m in range(4):
        print("Feature {}. median = {:.2f}".format(m,state[m]))
    plt.bar([0,1,2,3],state,width=0.5,label='Feature\'s Median value',color='green',tick_label=[0,1,2,3])
    plt.legend()
    plt.xlabel('feature')
    plt.ylabel('value')
    plt.show()
#找出中位數並畫圖(先分別把database以四個直行分別存取，再丟進來處理    
def foutput(value,species):
    file=r"d:\大學課業\大四\ML\HW2\0811002_Shwan_iris_data.csv"
    outfile=open(file,'w+')
    writein=csv.writer(outfile)
    for n in range(0,150,1):
        value[n].append(species[n])
    writein.writerows(value)
    outfile.close
    
    
    
value=[]    
for i in range(0,151,1):
    data=str(irisdata.readline())
    data = data.replace('[',' ')
    data = data.replace(']',' ')
    data = data.split()
    value.append(data)
value.remove(value[0])


specie = []
for i in range(num+2, num+8, 1):
    sep = str(irisdata.readline())
    sep = sep.split()
    specie.append(sep)
specie.remove(specie[0])
species=[]
for i in range(0,5,1):
    species.extend(specie[i])

number(species)
spec1=[]
spec2=[]
spec3=[]
spec4=[]
for j in range(0,150,1):
    spec1.append(value[j][0])
    spec2.append(value[j][1])
    spec3.append(value[j][2])
    spec4.append(value[j][3])
spec1=list(map(float,spec1))
spec2=list(map(float,spec2))
spec3=list(map(float,spec3))
spec4=list(map(float,spec4))
findmed(spec1,spec2,spec3,spec4)
np.random.shuffle(value)
np.random.shuffle(species)
foutput(value,species)
