import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt

e  = 1/np.power(10,5)
mi = 1000

# f(1)
x1 = []
x2 = []
x3 = []
pv = []
x = np.array([1, 1, 1])
BF1(x)
plt.plot(range(len(pv)),pv)
plt.ylabel('Gradient Norm')
plt.xlabel('Iteration')

# f(2)
x1 = []
x2 = []
pv = []
x = np.array([0,0])
BF2(x)
u = np.linspace(-.5,2,1000)
v = np.linspace(-.5,2,1000)
z1,z2 = np.meshgrid(u,v)
us = np.asarray(x1)
vs = np.asarray(x2)
z = np.power(z1,2)+2*np.power(z2,2)-2*z1*z2-2*z2
plt.contourf(z1,z2,z)
zs = np.power(us,2)+2*np.power(vs,2)-2*us*vs-2*vs
plt.plot(us,vs)

plt.plot(range(len(pv)),pv)
plt.ylabel('Gradient Norm')
plt.xlabel('Iteration')

# f(3)
x1 = []
x2 = []
pv = []
x = np.array([-1.2,1])
BF3(x)
u = np.linspace(-1.3,2,1000)
v = np.linspace(-2,3,1000)
z1,z2 = np.meshgrid(u,v)
us = np.asarray(x1)
vs = np.asarray(x2)
z = 100*np.power((z2-np.power(z1,2)),2)+np.power((1-z1),2)
plt.contourf(z1,z2,z)
zs = 100*np.power((vs-np.power(us,2)),2)+np.power((1-us),2)
plt.plot(us,vs)

plt.plot(range(len(pv)),pv)
plt.ylabel('Gradient Norm')
plt.xlabel('Iteration')

# f(4)
x1 = []
x2 = []
pv = []
x = np.array([2,-2])
BF4(x)
u = np.linspace(-1,2.5,100)
v = np.linspace(-2,2,100)
z1,z2 = np.meshgrid(u,v)
us = np.asarray(x1)
vs = np.asarray(x2)
z = np.power((z2+z1),4)+np.power(z2,2)
plt.contourf(z1,z2,z)
zs = np.power((vs+us),4)+np.power(vs,2)
plt.plot(us,vs)

plt.plot(range(len(pv)),pv)
plt.ylabel('Gradient Norm')
plt.xlabel('Iteration')

# f(5.1)
x1 = []
x2 = []
av = []
pv = []
c  = 1
cc = 1
x = np.array([1,-1])
BF5(x)
u = np.linspace(-1,1.5,1000)
v = np.linspace(-1,1.5,1000)
z1,z2 = np.meshgrid(u,v)
us = np.asarray(x1)
vs = np.asarray(x2)
z = np.power((z1-1),2)+np.power((z2-1),2)+c*np.power(np.power(z1,2)+np.power(z2,2)-.25,2)
plt.contourf(z1,z2,z)
zs = np.power((us-1),2)+np.power((vs-1),2)+c*np.power(np.power(us,2)+np.power(vs,2)-.25,2)
plt.plot(us,vs)

plt.plot(range(len(pv)),pv)
plt.ylabel('Gradient Norm')
plt.xlabel('Iteration')

# f(5.2)
x1 = []
x2 = []
av = []
pv = []
c  = 10
cc = 10
x = np.array([1,-1])
BF5(x)
u = np.linspace(-1,1.5,1000)
v = np.linspace(-1,1.5,1000)
z1,z2 = np.meshgrid(u,v)
us = np.asarray(x1)
vs = np.asarray(x2)
z = np.power((z1-1),2)+np.power((z2-1),2)+c*np.power(np.power(z1,2)+np.power(z2,2)-.25,2)
plt.contourf(z1,z2,z)
zs = np.power((us-1),2)+np.power((vs-1),2)+c*np.power(np.power(us,2)+np.power(vs,2)-.25,2)
plt.plot(us,vs)

plt.plot(range(len(pv)),pv)
plt.ylabel('Gradient Norm')
plt.xlabel('Iteration')

# f(5.3)
x1 = []
x2 = []
av = []
pv = []
c  = 100
cc = 100
x = np.array([1,-1])
BF5(x)
u = np.linspace(-1,1.5,1000)
v = np.linspace(-1,1.5,1000)
z1,z2 = np.meshgrid(u,v)
us = np.asarray(x1)
vs = np.asarray(x2)
z = np.power((z1-1),2)+np.power((z2-1),2)+c*np.power(np.power(z1,2)+np.power(z2,2)-.25,2)
plt.contourf(z1,z2,z)
zs = np.power((us-1),2)+np.power((vs-1),2)+c*np.power(np.power(us,2)+np.power(vs,2)-.25,2)
plt.plot(us,vs)

plt.plot(range(len(pv)),pv)
plt.ylabel('Gradient Norm')
plt.xlabel('Iteration')

######################################  1  #########################################

def f1(x):
    return np.power(x[0],2)+np.power(x[1],2)+np.power(x[2],2)

def df1(x):
    return np.array([2*x[0], 2*x[1], 2*x[2]])

def f1d(alpha,x,p):
    return f1(x + alpha*p)

def con1(x):
    return np.linalg.norm(df1(x))/(1+np.abs(f1(x)))

def ss(xo,xn):
    return xn-xo

def yy1(xo,xn):
    return df1(xn)-df1(xo)

def BB(B,y,s):
    Bs  = np.dot(B,s)
    BsT = np.transpose(Bs)
    yT  = np.transpose(y)
    return B - np.dot(Bs,BsT)/np.dot(np.transpose(s),Bs)+np.dot(y,yT)/np.dot(yT,s)

def pp(B,df):
    return np.dot(np.linalg.inv(B),(-1)*df)

def BF1(x):
    it=0
    flag=True
    B = np.identity(x.shape[0])
    while(flag):
        p = pp(B,df1(x))
        a = sopt.golden(f1d, args=(x,p,))
        xo = x
        xn = x + a * p
        s  = ss(xo,xn)
        y  = yy1(xo,xn)
        B  = BB(B,y,s)
        x = xn
        x1.append(x[0])
        x2.append(x[1])
        x3.append(x[2])
        pv.append(np.linalg.norm(df1(x)))
        it = it + 1
        print('it=',it,'\n')
        print('p=',p,'\n')
        print('a=',a,'\n')
        print('x=',x,'\n')
        if(con1(x)<e or it>=mi):
            flag = False
            print(it,'\n')
            return x

######################################  2  #########################################

def f2(x):
    return np.power(x[0],2)+2*np.power(x[1],2)-2*x[0]*x[1]-2*x[1]

def df2(x):
    return np.array([2*x[0]-2*x[1],4*x[1]-2*x[0]-2])

def f2d(alpha,x,p):
    return f2(x + alpha*p)

def con2(x):
    return np.linalg.norm(df2(x))/(1+np.abs(f2(x)))

def ss(xo,xn):
    return xn-xo

def yy2(xo,xn):
    return df2(xn)-df2(xo)

def BB(B,y,s):
    Bs  = np.dot(B,s)
    BsT = np.transpose(Bs)
    yT  = np.transpose(y)
    return B - np.dot(Bs,BsT)/np.dot(np.transpose(s),Bs)+np.dot(y,yT)/np.dot(yT,s)

def pp(B,df):
    return np.dot(np.linalg.inv(B),(-1)*df)
    # return np.linalg.lstsq(B,-df)[0]

def BF2(x):
    it=0
    flag=True
    B = np.identity(x.shape[0])
    while(flag):
        p = pp(B,df2(x))
        a = sopt.golden(f2d, args=(x,p,))
        xo = x
        xn = x + a * p
        s  = ss(xo,xn)
        y  = yy2(xo,xn)
        B  = BB(B,y,s)
        x = xn
        it = it + 1
        # print('it=',it,'\n')
        # print('p=',p,'\n')
        # print('a=',a,'\n')
        # print('x=',x,'\n')
        # print('B=',B,'\n')
        x1.append(x[0])
        x2.append(x[1])
        pv.append(np.linalg.norm(df2(x)))
        if(con2(x)<e or it>=mi):
            flag = False
            print(it,'\n')
            return x

######################################  3  #########################################

def f3(x):
    return 100*np.power((x[1]-np.power(x[0],2)),2)+np.power((1-x[0]),2)

def df3(x):
    return np.array([-400*x[0]*x[1]+400*np.power(x[0],3)-2+2*x[0],200*x[1]-200*np.power(x[0],2)])

def f3d(alpha,x,p):
    return f3(x + alpha*p)

def con3(x):
    return np.linalg.norm(df3(x))/(1+np.abs(f3(x)))

def ss(xo,xn):
    return xn-xo

def yy3(xo,xn):
    return df3(xn)-df3(xo)

def BB(B,y,s):
    Bs  = np.dot(B,s)
    BsT = np.transpose(Bs)
    yT  = np.transpose(y)
    return B - np.dot(Bs,BsT)/np.dot(np.transpose(s),Bs)+np.dot(y,yT)/np.dot(yT,s)

def pp(B,df):
    return np.dot(np.linalg.inv(B),(-1)*df)

def BF3(x):
    it=0
    flag=True
    B = np.identity(x.shape[0])
    while(flag):
        p = pp(B,df3(x))
        a = sopt.golden(f3d, args=(x,p,))
        xo = x
        xn = x + a * p
        s  = ss(xo,xn)
        y  = yy3(xo,xn)
        B  = BB(B,y,s)
        x = xn
        it = it + 1
        # print('it=',it,'\n')
        # print('p=',p,'\n')
        # print('a=',a,'\n')
        # print('x=',x,'\n')
        x1.append(x[0])
        x2.append(x[1])
        pv.append(np.linalg.norm(df3(x)))
        if(con3(x)<e or it>=mi):
            flag = False
            print(it,'\n')
            return x

######################################  4  #########################################

def f4(x):
    return np.power((x[1]+x[0]),4)+np.power(x[1],2)

def df4(x):
    t = 4*np.power(x[0],3)+12*np.power(x[0],2)*x[1]+12*x[0]*np.power(x[1],2)+4*np.power(x[1],3)
    return np.array([t,t+2*x[1]])

def f4d(alpha,x,p):
    return f4(x + alpha*p)

def con4(x):
    return np.linalg.norm(df4(x))/(1+np.abs(f4(x)))

def ss(xo,xn):
    return xn-xo

def yy4(xo,xn):
    return df4(xn)-df4(xo)

def BB(B,y,s):
    Bs  = np.dot(B,s)
    BsT = np.transpose(Bs)
    yT  = np.transpose(y)
    return B - np.dot(Bs,BsT)/np.dot(np.transpose(s),Bs)+np.dot(y,yT)/np.dot(yT,s)

def pp(B,df):
    return np.dot(np.linalg.inv(B),(-1)*df)

def BF4(x):
    it=0
    flag=True
    B = np.identity(x.shape[0])
    while(flag):
        p = pp(B,df4(x))
        a = sopt.golden(f4d, args=(x,p,))
        xo = x
        xn = x + a * p
        s  = ss(xo,xn)
        y  = yy4(xo,xn)
        B  = BB(B,y,s)
        x = xn
        it = it + 1
        # print('it=',it,'\n')
        # print('p=',p,'\n')
        # print('a=',a,'\n')
        # print('x=',x,'\n')
        x1.append(x[0])
        x2.append(x[1])
        pv.append(np.linalg.norm(df4(x)))
        if(con4(x)<e or it>=mi):
            flag = False
            print(it,'\n')
            return x

######################################  5  #########################################

def f5(x):
    return np.power((x[0]-1),2)+np.power((x[1]-1),2)+c*np.power(np.power(x[0],2)+np.power(x[1],2)-.25,2)

def df5(x):
    return np.array([2*x[0]-2+4*c*np.power(x[0],3)+4*c*x[0]*np.power(x[1],2)-c*x[0],2*x[1]-2+4*c*np.power(x[0],2)*x[1]+4*c*np.power(x[1],3)-c*x[1]])

def f5d(alpha,x,p):
    return f5(x + alpha*p)

def con5(x):
    return np.linalg.norm(df5(x))/(1+np.abs(f5(x)))

def ss(xo,xn):
    return xn-xo

def yy5(xo,xn):
    return df5(xn)-df5(xo)

def BB(B,y,s):
    Bs  = np.dot(B,s)
    BsT = np.transpose(Bs)
    yT  = np.transpose(y)
    return B - np.dot(Bs,BsT)/np.dot(np.transpose(s),Bs)+np.dot(y,yT)/np.dot(yT,s)

def pp(B,df):
    return np.dot(np.linalg.inv(B),(-1)*df)

def BF5(x):
    it=0
    flag=True
    B = np.identity(x.shape[0])
    while(flag):
        p = pp(B,df5(x))
        a = sopt.golden(f5d, args=(x,p,))
        xo = x
        xn = x + a * p
        s  = ss(xo,xn)
        y  = yy5(xo,xn)
        B  = BB(B,y,s)
        x = xn
        it = it + 1
        # print('it=',it,'\n')
        # print('p=',p,'\n')
        # print('a=',a,'\n')
        # print('x=',x,'\n')
        x1.append(x[0])
        x2.append(x[1])
        pv.append(np.linalg.norm(df5(x)))
        if(con5(x)<e or it>=mi):
            flag = False
            print(it,'\n')
            return x




## Counter plot
u = np.linspace(-1,1,1000)
v = np.linspace(-2,2,1000)
z1,z2 = np.meshgrid(u,v)
us = np.linspace(-1,1,10)
vs = np.linspace(-1,1,10)
z = np.power(z1,2)+np.power(z2,2)
plt.contourf(z1,z2,z)
zs = np.power(us,2)+np.power(vs,2)
plt.plot(us,vs)



