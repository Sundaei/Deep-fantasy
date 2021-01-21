def OR_gate(i,j):
    w1=0.6;w2=0.6;b=-0.5
    result=i*w1+j*w2+b
    if result <=0:
        return 0
    else:
        return 1

def NAND_gate(i,j):
    w1=-0.5;w2=-0.5;b=0.7
    result=i*w1+j*w2+b
    if result <=0:
        return 0
    else:
        return 1

def AND_gate(i,j):
    w1=0.5;w2=0.5;b=-0.7
    result=i*w1+j*w2+b
    if result <=0:
        return 0
    else:
        return 1

def XOR_gate(x1,x2):
    s1=NAND_gate(x1,x2)
    s2=OR_gate(x1,x2)
    return AND_gate(s1,s2)
print(AND_gate(1,1))
print(XOR_gate(0,0))