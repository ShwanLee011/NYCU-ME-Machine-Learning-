file = "d:\大學課業\大四\機器學習原理與工業應用 Machine Learning\primenum.txt"
outfile = open(file,'r')
lines = str(outfile.read())
num = lines.split()

count = 0
for n in num:
    outnum = float(n)
    outnum = int(outnum)
    if(outnum > 3000) and (outnum < 6000) :
        count+=1
print("I, Shwan Lee, 0811002, found "+str(count)+" prime numbers between 3000 and 6000.")
