import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from pylab import *
file = "d:\data1.txt"
database = open(file,'r')
def readf():
    readin = []
    for i in range(6):
        data = database.readline()
        data = data.replace('[',' ')
        data = data.replace(']',' ')
        data = data.split()
        readin.append(data)
    readin = np.array(readin)
    return readin

def rotmatrix(angle):
    rad = angle*(math.pi)/180
    rmat = []
    r00 = math.cos(rad)
    r01 = -1*math.sin(rad)
    r10 = math.sin(rad)
    r11 = math.cos(rad)
    rmat.append(r00)
    rmat.append(r01)
    rmat.append(r10)
    rmat.append(r11)
    rmat1 = np.array(rmat)
    rmat1 = rmat1.reshape(-1,2)
    print(rmat1)
    return rmat
def rotate(datain,rotmat):
    outputx = []
    outputy = []
    result = []
    for j in range(0,6,1):
        a = rotmat[0]*float(datain[j][0])+rotmat[1]*float(datain[j][1])
        b = rotmat[2]*float(datain[j][0])+rotmat[3]*float(datain[j][1])
        outputx.append(a)
        outputy.append(b)
        result.append("{:.3f}".format(a))
        result.append("{:.3f}".format(b))
    result = np.array(result)
    result = result.reshape(-1,2)
    return result
def plotresult(result):
    resultx = []
    resulty = []
    for k in range(6):
        resultx.append(float(result[k][0]))
        resulty.append(float(result[k][1]))
    resultx = np.array(resultx)
    resulty = np.array(resulty)
    plt.plot(resultx, resulty, marker='o', linewidth = 1, markersize = 15, color = 'r')
    plt.bar(resultx, resulty, width = 0.05, label = 'After Rotation', color = 'green', tick_label = resultx)
    plt.legend()
    plt.xlabel('X coord.')
    plt.ylabel('Y coord.')
    plt.show()
def writefile(result):
    out = r"d:\frame0.csv"
    outfile = open(out,'w')
    writein = csv.writer(outfile)
    result = np.array(result)
    writein.writerows(result)
    outfile.close
datain = readf()
print("Values before rotation is :")
print(datain)
s1 = input("Enter the rotation angle (in degrees) : ")
angle = eval(s1)
rotmat = rotmatrix(angle)
result = []
result = rotate(datain,rotmat)
print('Values after rotation are: ')
print(result)
writefile(result)
plotresult(result)
