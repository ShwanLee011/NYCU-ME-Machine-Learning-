import math
import numpy as np

lower_lim=200

upper_lim=9900

file = "d:\大學課業\大四\機器學習原理與工業應用 Machine Learning\primenum.txt"
data = []
array1 = np.array(data)
count = 0
for i in range(lower_lim, upper_lim, 1):
    for j in range(2, int(math.sqrt(upper_lim)+1),1):
        if(i%j==0):
            break #只要出現被整出就跳開避免誤寫
        elif(j==int(math.sqrt(upper_lim))):
            array1 = np.append(array1,int(i))
            count+=1
output = np.flip(array1) #重新排序(由後往前)
outfile = open(file,'w+')
for n in range(1,count+1,1):
    if (n%6==0):
        outfile.write(str(output[n-1]))  #轉成字串輸入檔案內，逢6換一行
        outfile.write(' ')
        outfile.write('\n')
    else:
        outfile.write(str(output[n-1]))
        outfile.write(" ")
outfile.close( )


#這邊不用窮舉法，改用運算較少的寫法：因為100的平方根為10，所以100以內的
#數字只要不能被小於10的質數整除，就一定是質數，同理10000以內的只要找100
#以內的除數就好
