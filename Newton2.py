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
x = np.matrix('1; 1; 1')
NM1(x,e,mi)
plt.plot(range(len(pv)),pv)
plt.ylabel('Gradient Norm')
plt.xlabel('Iteration')

# f(2)
x1 = []
x2 = []
x3 = []
pv = []
x = np.matrix('0 ; 0')
NM2(x,e,mi)
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
x3 = []
pv = []
x = np.matrix('-1.2 ; 1')
NM3(x,e,mi)

u = np.linspace(-1.3,2,1000)
v = np.linspace(-4,3,1000)
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
av = []
pv = []
x = np.matrix('2 ; -2')
NM4(x,e,mi)
u = np.linspace(-3,3,100)
v = np.linspace(-3,3,100)
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

# f(5_1)
x1 = []
x2 = []
av = []
pv = []
c  = 1
cc = 1
xt = np.matrix('2 ; -2')
NM5(xt,e,mi)

u = np.linspace(-2,2.5,1000)
v = np.linspace(-3,1.5,1000)
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

# f(5_2)
x1 = []
x2 = []
av = []
pv = []
c  = 10
cc  = 10
xt = np.matrix('2 ; -2')
NM5(xt,e,mi)

u = np.linspace(-2,2.5,1000)
v = np.linspace(-3,1.5,1000)
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

# f(5_3)
x1 = []
x2 = []
av = []
pv = []
c  = 100
cc = 100
xt = np.matrix('2 ; -2')
NM5(xt,e,mi)

u = np.linspace(-2,2.5,1000)
v = np.linspace(-3,1.5,1000)
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

def NM1(x,e,mi):
    fg = True
    it = 0
    print('x0 = \n',x,'\n')
    while(fg == True):
        Df = gdnm(x)
        x1.append(x[0])
        x2.append(x[1])
        x3.append(x[2])
        pv.append(np.linalg.norm(Df))
        it = it + 1
        if(checknm(x,Df,e) or it>mi):
            if (it>mi):
                print('Solution found after 1000 iterations \n')
            return print('x* =\n',x)
            fg = False
        else:
            x = nxnm(x,Df)
            if(it<=10):
                print('x',it,'=','\n',x,'\n')

def gdnm(xo):
    df = np.zeros((3,1))
    x1 = xo[0,0]
    x2 = xo[1,0]
    x3 = xo[2,0]
    df[0,0] = 2*x1
    df[1,0] = 2*x2
    df[2,0] = 2*x3
    return df

def ssnm(xo):
    d2f = 2*np.identity(3)
    return np.transpose(np.linalg.inv(d2f))

def fnm(xo):
    x1 = xo[0, 0]
    x2 = xo[1, 0]
    x3 = xo[2, 0]
    return np.power(x1,2)+np.power(x2,2)+np.power(x3,2)

def checknm(x,Df,e):
    n = np.linalg.norm(Df,2)
    d = 1 + np.abs(fnm(x))
    v = n/d
    return np.asscalar(v) < e

def nxnm(xo,Df):
    return xo-np.dot(ssnm(xo),Df)

######################################  2  #########################################

def NM2(x,e,mi):
    fg = True
    it = 0
    print('x0 = \n',x,'\n')
    while(fg == True):
        Df = gdnm2(x)
        x1.append(np.asscalar(x[0]))
        x2.append(np.asscalar(x[1]))
        pv.append(np.linalg.norm(Df))
        it = it + 1
        if(checknm2(x,Df,e) or it>mi):
            if (it>mi):
                print('Solution found after 1000 iterations \n')
            return print('x* =\n',x)
            fg = False
        else:
            x = nxnm2(x,Df)
            if(it<=10):
                print('x',it,'=','\n',x,'\n')

def gdnm2(xo):
    df = np.zeros((2,1))
    x1 = xo[0,0]
    x2 = xo[1,0]
    df[0,0] = 2*x1-2*x2
    df[1,0] = 4*x2-2*x1-2
    return df

def ssnm2(xo):
    d2f = np.matrix('2 -2 ; -2 4')
    return np.transpose(np.linalg.inv(d2f))

def fnm2(xo):
    x1 = xo[0, 0]
    x2 = xo[1, 0]
    return np.power(x1,2)+2*np.power(x2,2)-2*x1*x2-2*x2

def checknm2(x,Df,e):
    n = np.linalg.norm(Df,2)
    d = 1 + np.abs(fnm2(x))
    v = n/d
    return np.asscalar(v) < e

def nxnm2(xo,Df):
    return xo-np.dot(ssnm2(xo),Df)

######################################  3n  #########################################

def NM3(x,e,mi):
    fg = True
    it = 0
    print('x0 = \n',x,'\n')
    while(fg == True):
        Df = gdnm3(x)
        x1.append(np.asscalar(x[0]))
        x2.append(np.asscalar(x[1]))
        pv.append(np.linalg.norm(Df))
        it = it + 1
        if(checknm3(x,Df,e) or it>mi):
            if (it>mi):
                print('Solution found after 1000 iterations \n')
            return print('x* =\n',x)
            fg = False
        else:
            x = nxnm3(x,Df)
            if(it<=10):
                print('x',it,'=','\n',x,'\n')

def gdnm3(xo):
    x1 = xo[0, 0]
    x2 = xo[1, 0]
    df = np.zeros((2, 1))
    df[0, 0] = (-400)*x1*x2+400*np.power(x1,3)+2*x1-2
    df[1, 0] = 200*x2-200*np.power(x1,2)
    return df

def ssnm3(xo):
    d2f = np.zeros((2,2))
    x1 = xo[0, 0]
    x2 = xo[1, 0]
    d2f[0,0] = (-400)*x2+1200*np.power(x1,2)+2
    d2f[1,0] = (-400)*x1
    d2f[0,1] = (-400)*x1
    d2f[1,1] = 200
    return np.transpose(np.linalg.inv(d2f))

def fnm3(xo):
    x1 = xo[0, 0]
    x2 = xo[1, 0]
    return 100*np.power((x2-np.power(x1,2)),2)+np.power((1-x1),2)

def checknm3(x,Df,e):
    n = np.linalg.norm(Df,2)
    d = 1 + np.abs(fnm3(x))
    v = n/d
    return np.asscalar(v) < e

def nxnm3(xo,Df):
    return xo-np.dot(ssnm3(xo),Df)

######################################  4n  #########################################

def NM4(x,e,mi):
    fg = True
    it = 0
    print('x0 = \n',x,'\n')
    while(fg == True):
        Df = gdnm3(x)
        x1.append(np.asscalar(x[0]))
        x2.append(np.asscalar(x[1]))
        pv.append(np.linalg.norm(Df))
        it = it + 1
        if(checknm4(x,Df,e) or it>mi):
            if (it>mi):
                print('Solution found after 1000 iterations \n')
            return print('x* =\n',x)
            fg = False
        else:
            x = nxnm4(x,Df)
            if(it<=10):
                print('x',it,'=','\n',x,'\n')

def gdnm4(xo):
    x1 = xo[0, 0]
    x2 = xo[1, 0]
    df = np.zeros((2, 1))
    df[0, 0] = 4*np.power(x1,3)+12*np.power(x1,2)*x2+12*x1*np.power(x2,2)+4*np.power(x2,3)
    df[1, 0] = 4*np.power(x1,3)+12*np.power(x1,2)*x2+12*x1*np.power(x2,2)+4*np.power(x2,3)+2*x2
    return df

def ssnm4(xo):
    d2f = np.zeros((2,2))
    x1 = xo[0, 0]
    x2 = xo[1, 0]
    d2f[0,0] = 12*np.power(x1,2)+24*x1*x2+12*np.power(x2,2)
    d2f[1,0] = 12*np.power(x1,2)+24*x1*x2+12*np.power(x2,2)
    d2f[0,1] = 12*np.power(x1,2)+24*x1*x2+12*np.power(x2,2)
    d2f[1,1] = 12*np.power(x1,2)+24*x1*x2+12*np.power(x2,2)+2
    return np.transpose(np.linalg.inv(d2f))

def fnm4(xo):
    x1 = xo[0, 0]
    x2 = xo[1, 0]
    return np.power((x1+x2),4)+np.power(x2,2)

def checknm4(x,Df,e):
    n = np.linalg.norm(Df,2)
    d = 1 + np.abs(fnm4(x))
    v = n/d
    return np.asscalar(v) < e

def nxnm4(xo,Df):
    return xo-np.dot(ssnm4(xo),Df)

######################################  5n  #########################################

def NM5(x,e,mi):
    fg = True
    it = 0
    print('x0 = \n',x,'\n')
    while(fg == True):
        Df = gdnm5(x)
        x1.append(np.asscalar(x[0]))
        x2.append(np.asscalar(x[1]))
        pv.append(np.linalg.norm(Df))
        it = it + 1
        if(checknm5(x,Df,e) or it>mi):
            if (it>mi):
                print('Solution found after 1000 iterations \n')
            return print('x* =\n',x)
            fg = False
        else:
            x = nxnm5(x,Df)
            if(it<=10):
                print('x',it,'=','\n',x,'\n')

def gdnm5(xo):
    x1 = xo[0,0]
    x2 = xo[1,0]
    df = np.zeros((2,1))
    df[0,0] = 2*x1-2+4*x1*cc*(np.power(x1,2)+np.power(x2,2)-0.25)
    df[1,0] = 2*x2-2+4*x2*cc*(np.power(x1,2)+np.power(x2,2)-0.25)
    return df

def ssnm5(xo):
    d2f = np.zeros((2,2))
    x1 = xo[0, 0]
    x2 = xo[1, 0]
    d2f[0,0] = 2+12*cc*np.power(x1,2)+4*cc*np.power(x2,2)-cc
    d2f[1,0] = 8*cc*x1*x2
    d2f[0,1] = 8*cc*x1*x2
    d2f[1,1] = 2+4*cc*np.power(x1,2)+12*cc*np.power(x2,2)-cc
    return np.transpose(np.linalg.inv(d2f))

def fnm5(xo):
    x1 = xo[0, 0]
    x2 = xo[1, 0]
    return np.power((x1-1),2)+np.power((x2-1),2)+cc*np.power((np.power(x1,2)+np.power(x2,2)-0.25),2)

def checknm5(x,Df,e):
    n = np.linalg.norm(Df,2)
    d = 1 + np.abs(fnm5(x))
    v = n/d
    return np.asscalar(v) < e

def nxnm5(xo,Df):
    return xo-np.dot(ssnm5(xo),Df)






