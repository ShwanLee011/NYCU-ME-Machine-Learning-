import math

def eqnsolver(a,b,c):
    if(b*b-4*a*c > 0):
        root_1=(-b+math.sqrt(b*b-4*a*c))/(2*a)
        root_2=(-b-math.sqrt(b*b-a*a*c))/(2*a)
        print("2 real roots, x1={:.1f}, x2={:.1f}".format(root_1,root_2))
    elif(b*b-4*a*c == 0):
        root=-b/(2*a)
        print("2 same root, x={:.1f}".format(root))
    elif(b*b-4*a*c < 0):
        root_realpart=-b/(2*a)
        root_imagepart=math.sqrt(4*a*c-b*b)/(2*a)
        print("2 image roots, x1={:.1f} + {:.1f} J, x2={:.1f} - {:.1f} J ".format(root_realpart,root_imagepart,root_realpart,root_imagepart))
 
print("Solving 2nd order eqn a X^2+bX+c=0")
s1=input("a ")
a=int(s1)
if(a!=0):
    s2=input("b ")
    b=int(s2)
    s3=input("c ")
    c=int(s3)
    eqnsolver(a,b,c)

