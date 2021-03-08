import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

def p_fun(x):
	return (np.exp(-x/2))/(2*(1-np.exp(-2)))
	
def p_fun_2(x):
	return 65*(np.cos(2*x)*np.cos(2*x)*np.exp(-x/2))/(66*(1-np.exp(-np.pi/2)))	

def p_fun_2_1(x):
	return (np.exp(-x/2))/(2*(1-np.exp(-np.pi/2)))

def F_odwr_fun(u):
	C=(1-np.exp(-2))
	return 2*np.log(1/(1-C*u))

def F_odwr_fun_2(u):
	C=(1-np.exp(-np.pi/2))
	return 2*np.log(1/(1-C*u))

def gen_fun(n):
	tab=rng.uniform(0,1,N)
	tab=F_odwr_fun(tab)
	return tab
#################
N=1000000
rng = default_rng()
#################

tab=gen_fun(N)
tab = tab[~np.isnan(tab)]
print(len(tab))
plt.hist(x=tab,bins=40,density=True,label="generowane przypadki")
x=np.linspace(0,4,50)
plt.plot(x,p_fun(x),label="funkcja f(x)")
plt.legend(frameon=False)
plt.show()
#################
C=(65/66)*2
x2=np.linspace(0,np.pi,50)
tab2=[]

print(len(tab2))
while len(tab2)<1000000:
	xtry=rng.uniform(0,1)
	xtry=F_odwr_fun_2(xtry)
	u=rng.uniform(0,1)
	if (p_fun_2(xtry)>p_fun_2_1(xtry)*C*u):
		tab2.append(xtry)
	
# print(tab2)

plt.axes()
plt.hist(x=tab2,bins='auto',density=True,label="Generowane przypadki")
plt.plot(x2,C*p_fun_2_1(x2),label="c*g(x)")
plt.plot(x2,p_fun_2(x2),label="f(x)")
plt.legend(frameon=False)
plt.show()