import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
e  = 1/np.power(10,5)
mi = 1000

####################### Conjugate Gradient ########################
########################                   ########################

# f(1)
x1 = []
x2 = []
x3 = []
pv = []
x = np.array([1, 1, 1])
CG1(x)
plt.plot(range(len(pv)),pv)
plt.ylabel('Gradient Norm')
plt.xlabel('Iteration')

# f(2)
x1 = []
x2 = []
pv = []
x = np.array([0,0])
CG2(x)
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
CG3(x)
u = np.linspace(-2,2,1000)
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
CG4(x)
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
pv = []
c  = 1
cc = 1
x = np.array([1,-1])
CG5(x)
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
pv = []
c  = 10
cc = 10
x = np.array([1,-1])
CG5(x)
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
pv = []
c  = 100
cc = 100
x = np.array([1,-1])
CG5(x)
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


#####################################  1  #####################################

def f1(x):
    return np.power(x[0],2)+np.power(x[1],2)+np.power(x[2],2)

def df1(x):
    return np.array([2*x[0], 2*x[1], 2*x[2]])

def grad1(xo):
    return 2*np.identity(3)

def a1(d0,df,Q):
    return (-1)*np.dot(np.transpose(d0),df)/np.dot(np.transpose(d0),np.dot(Q,d0))

def beta(d0,df,Q):
    return np.dot(-np.transpose(d0),np.dot(Q,df))/np.dot(np.transpose(d0),np.dot(Q,d0))

def d(df,B,d0):
    return -df+B*d0

def con1(x):
    return np.linalg.norm(df1(x))/(1+np.abs(f1(x)))

def CG1(x):
    it = 0
    flag = True
    while(flag):
        if(it==0):
            d0 = df1(x)
            Q  = grad1(x)
            a  = a1(d0, d0, Q)
        else:
            Q = grad1(x)
            a = a1(d0, df1(x), Q)
        xn = x + a*d0
        df1n = df1(xn)
        B  = beta(d0,df1n,Q)
        x = xn
        d0 = d(df1n,B,d0)
        it = it+1
        print('it=', it, '\n')
        print('a=', a, '\n')
        print('x=', x, '\n')
        x1.append(x[0])
        x2.append(x[1])
        x3.append(x[2])
        pv.append(np.linalg.norm(df1(x)))
        if (con1(x) < e or it >= mi):
            flag = False
            print(it, '\n')
            return x


#####################################  2  #####################################

def f2(x):
    return np.power(x[0],2)+2*np.power(x[1],2)-2*x[0]*x[1]-2*x[1]

def df2(x):
    return np.array([2*x[0]-2*x[1],4*x[1]-2*x[0]-2])

def grad2(x):
    return np.array([[2,-2],[-2,4]])

def a2(d0,df,Q):
    return (-1)*np.dot(np.transpose(d0),df)/np.dot(np.transpose(d0),np.dot(Q,d0))

def beta(d0,df,Q):
    return np.dot(-np.transpose(d0),np.dot(Q,df))/np.dot(np.transpose(d0),np.dot(Q,d0))

def d(df,B,d0):
    return -df+B*d0

def con2(x):
    return np.linalg.norm(df2(x))/(1+np.abs(f2(x)))

def CG2(x):
    it = 0
    flag = True
    while(flag):
        if(it==0):
            d0 = df2(x)
            Q  = grad2(x)
            a  = a2(d0, d0, Q)
        else:
            Q = grad2(x)
            a = a2(d0, df2(x), Q)
        xn = x + a*d0
        df1n = df2(xn)
        B  = beta(d0,df1n,Q)
        x = xn
        d0 = d(df1n,B,d0)
        it = it+1
        # print('it=', it, '\n')
        # print('a=', a, '\n')
        # print('x=', x, '\n')
        x1.append(x[0])
        x2.append(x[1])
        pv.append(np.linalg.norm(df2(x)))
        if (con2(x) < e or it >= mi):
            flag = False
            print(it, '\n')
            return x

#####################################  3  #####################################

def f3(x):
    return 100*np.power((x[1]-np.power(x[0],2)),2)+np.power((1-x[0]),2)

def df3(x):
    return np.array([-400*x[0]*x[1]+400*np.power(x[0],3)-2+2*x[0],200*x[1]-200*np.power(x[0],2)])

def grad3(x):
    return np.array([[-400*x[1]+1200*np.power(x[0],2)+2,-400*x[0]],[-400*x[0],200]])

def a3(d0,df,Q):
    return (-1)*np.dot(np.transpose(d0),df)/np.dot(np.transpose(d0),np.dot(Q,d0))

def beta(d0,df,Q):
    return np.dot(-np.transpose(d0),np.dot(Q,df))/np.dot(np.transpose(d0),np.dot(Q,d0))

def d(df,B,d0):
    return -df+B*d0

def con3(x):
    return np.linalg.norm(df3(x))/(1+np.abs(f3(x)))

def CG3(x):
    it = 0
    flag = True
    while(flag):
        if(it==0):
            d0 = df3(x)
            Q  = grad3(x)
            a  = a3(d0, d0, Q)
        else:
            Q = grad3(x)
            a = a3(d0, df3(x), Q)
        xn = x + a*d0
        df1n = df3(xn)
        B  = beta(d0,df1n,Q)
        x = xn
        d0 = d(df1n,B,d0)
        it = it+1
        # print('it=', it, '\n')
        # print('a=', a, '\n')
        # print('x=', x, '\n')
        x1.append(x[0])
        x2.append(x[1])
        pv.append(np.linalg.norm(df3(x)))
        if (con3(x) < e or it >= mi):
            flag = False
            print(it, '\n')
            return x

#####################################  4  #####################################

def f4(x):
    return np.power((x[1]+x[0]),4)+np.power(x[1],2)

def df4(x):
    t = 4*np.power(x[0],3)+12*np.power(x[0],2)*x[1]+12*x[0]*np.power(x[1],2)+4*np.power(x[1],3)
    return np.array([t,t+2*x[1]])

def grad4(x):
    t1 = 12*np.power(x[0],2)+24*x[0]*x[1]+12*np.power(x[1],2)
    t2 = 12*np.power(x[0],2)+24*x[0]*x[1]+12*np.power(x[1],2)+2
    return np.array([[t1,t1],[t1,t2]])

def a4(d0,df,Q):
    return (-1)*np.dot(np.transpose(d0),df)/np.dot(np.transpose(d0),np.dot(Q,d0))

def beta(d0,df,Q):
    return np.dot(-np.transpose(d0),np.dot(Q,df))/np.dot(np.transpose(d0),np.dot(Q,d0))

def d(df,B,d0):
    return -df+B*d0

def con4(x):
    return np.linalg.norm(df4(x))/(1+np.abs(f4(x)))

def CG4(x):
    it = 0
    flag = True
    while(flag):
        if(it==0):
            d0 = df4(x)
            Q  = grad4(x)
            a  = a4(d0, d0, Q)
        else:
            Q = grad4(x)
            a = a4(d0, df4(x), Q)
        xn = x + a*d0
        df1n = df4(xn)
        B  = beta(d0,df1n,Q)
        x = xn
        d0 = d(df1n,B,d0)
        it = it+1
        # print('it=', it, '\n')
        # print('a=', a, '\n')
        # print('x=', x, '\n')
        x1.append(x[0])
        x2.append(x[1])
        pv.append(np.linalg.norm(df4(x)))
        if (con4(x) < e or it >= mi):
            flag = False
            print(it, '\n')
            return x

#####################################  5  #####################################

def f5(x):
    return np.power((x[0]-1),2)+np.power((x[1]-1),2)+c*np.power(np.power(x[0],2)+np.power(x[1],2)-.25,2)

def df5(x):
    return np.array([2*x[0]-2+4*c*np.power(x[0],3)+4*c*x[0]*np.power(x[1],2)-c*x[0],2*x[1]-2+4*c*np.power(x[0],2)*x[1]+4*c*np.power(x[1],3)-c*x[1]])

def grad5(x):
    t1 = 2+12*c*np.power(x[0],2)+4*c*np.power(x[1],2)-c
    t2 = 8*c*x[0]*x[1]
    t3 = 2+4*c*np.power(x[0],2)+12*c*np.power(x[1],2)-c
    return np.array([[t1,t2],[t2,t3]])

def a5(d0,df,Q):
    return (-1)*np.dot(np.transpose(d0),df)/np.dot(np.transpose(d0),np.dot(Q,d0))

def beta(d0,df,Q):
    return np.dot(-np.transpose(d0),np.dot(Q,df))/np.dot(np.transpose(d0),np.dot(Q,d0))

def d(df,B,d0):
    return -df+B*d0

def con5(x):
    return np.linalg.norm(df5(x))/(1+np.abs(f5(x)))

def CG5(x):
    it = 0
    flag = True
    while(flag):
        if(it==0):
            d0 = df5(x)
            Q  = grad5(x)
            a  = a5(d0, d0, Q)
        else:
            Q = grad5(x)
            a = a5(d0, df5(x), Q)
        xn = x + a*d0
        df1n = df5(xn)
        B  = beta(d0,df1n,Q)
        x = xn
        d0 = d(df1n,B,d0)
        it = it+1
        # print('it=', it, '\n')
        # print('a=', a, '\n')
        # print('x=', x, '\n')
        x1.append(x[0])
        x2.append(x[1])
        pv.append(np.linalg.norm(df5(x)))
        if (con5(x) < e or it >= mi):
            flag = False
            print(it, '\n')
            return x




