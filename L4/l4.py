import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from symfit import parameters, variables, sin, cos, Fit
import operator
import statistics
from scipy.stats import chisquare
import time
from mpl_toolkits import mplot3d
from matplotlib.ticker import LinearLocator


def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = 0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

def fun_gamma_k(x,k,mu):
	n=len(x)
	gamma=0
	for i in range(k,n):
		gamma+=(x[i]-mu)*(x[i-k]-mu)

	gamma=gamma/n
	return gamma

# def wielomian()

file="co2mlo.txt"
log="log.txt"
file_l=open(log,"w")

data=np.loadtxt(file,usecols=np.arange(0,7))
data=data[-12*20:,:]
x_data=data[:,2]
y_data=data[:,4]


x_data_f=data[:,2]
y_data_f=data[:,4]


############################## POLYN ##############################

polyfit=np.polyfit(x_data,y_data,3)
p=np.poly1d(polyfit)
file_l.write("Fit wielomianu=\n"+str((p)))

############################## FOURIER #################################
x, y = variables('x, y')
# w, = parameters('w')
w=1*2*np.pi
n=2
model_dict = {y: fourier_series(x, f=w, n=n)}

file_l.write("\nFit fouriera\n\n"+str(model_dict))

# Define a Fit object for this model and data
fit = Fit(model_dict, x=x_data, y=(y_data-p(x_data)))
fit_result = fit.execute()
file_l.write(str(fit_result))


# Plot the result
fourier_fit=fit.model(x=x_data, **fit_result.params).y


###############################	PLOT ##########################################
fig, ((ax1, ax2,ax3, ax4)) = plt.subplots(4, 1)

ax1.plot(x_data,y_data,label="data")
ax1.plot(x_data,p(x_data),)
ax1.set_title("dane + fit wielomianu")


ax2.plot(x_data,y_data-p(x_data))
ax2.set_title("dane po odjęciu fitu wielomianu")

ax3.plot(x_data,fourier_fit )
ax3.set_title("tylko fit fouriera")

ax4.plot(x_data,p(x_data))
ax4.plot(x_data,y_data-fourier_fit)
ax4.set_title("fit wielomian oraz dane minus fit fouriera")

fig.tight_layout()
fig.savefig("data_poly.png",dpi=500,pad_inches = 0)
plt.close()

fig, ((ax1, ax2)) = plt.subplots(2, 1)
ax1.plot(x_data,y_data-fourier_fit-p(x_data))
ax1.set_title("szereg")
ax2.plot(x_data,fourier_fit )
ax2.set_title("tylko fit fouriera")

fig.tight_layout()
fig.savefig("data_szum.png",dpi=500,pad_inches = 0)
plt.close()


fig, ((ax1)) = plt.subplots(1, 1)
x_data_przew=x_data

x_data_przew=np.concatenate((x_data_przew,(x_data+20)))
x_data_przew=x_data
# print(x_data_przew)
ax1.plot(x_data,y_data,label="dane")
fourier_fit2=fit.model(x=x_data_przew, **fit_result.params).y
ax1.plot(x_data_przew,p(x_data_przew),label="wielomian")
ax1.plot(x_data_przew,p(x_data_przew)+fourier_fit2,"--", label="wielomian + okresowy")
ax1.legend()
fig.savefig("przew.png",dpi=500,pad_inches = 0)

plt.close()

fig, ((ax1)) = plt.subplots(1, 1)
ax1.plot(x_data,y_data-fourier_fit-p(x_data))
ax1.set_title("Szereg reszt")
fig.savefig("szereg_reszt.png",dpi=500,pad_inches = 0)


fig, ((ax1)) = plt.subplots(1, 1)
ax1.plot(x_data,p(x_data),label="trend wielomianowy")
ax1.plot(x_data,y_data-fourier_fit,label="dane skorygowane sezonowo")
ax1.legend()
ax1.set_title("Dane+wielo")
fig.savefig("wielo_skorygowane_sez.png",dpi=500,pad_inches = 0)

#######################################
data_szereg=y_data-fourier_fit-p(x_data)
data_szereg2=np.random.uniform(-2,2,len(x_data))

gamma_0=fun_gamma_k(data_szereg,0,np.average(data_szereg))
gamma_0_2=fun_gamma_k(data_szereg2,0,np.average(data_szereg2))


sgm_dat=np.zeros(31)
sgm_dat_2=np.zeros(31)
for i in range(31):
	sgm_dat[i]=fun_gamma_k(data_szereg,i,np.average(data_szereg))/gamma_0
	sgm_dat_2[i]=fun_gamma_k(data_szereg2,i,np.average(data_szereg2))/gamma_0_2

fig, ((ax1)) = plt.subplots(1, 1)
ax1.plot(range(31),sgm_dat,label="dane")
ax1.plot(range(31),sgm_dat_2,label="szum")
ax1.set_title("fun autokorelacji")
ax1.legend()
fig.savefig("fun_auto.png",dpi=500,pad_inches = 0)

##############################################################################
##############################################################################

N=2000
phi_1=0.95
phi_2=-0.8
offset=20
N_last=100
k_numb=20

x_stoch=np.zeros(N+offset+2)
noise=np.zeros(N+offset+2)

for i in range(2,N+2+offset):
	eps=np.random.normal(0,0.3)
	x_stoch[i]=phi_1*x_stoch[i-1]+phi_2*x_stoch[i-2]+eps
	noise[i]=eps

fig, ((ax1)) = plt.subplots(1, 1)
ax1.plot(range(N_last),x_stoch[-N_last:],label="szereg - xi")
ax1.plot(range(N_last),noise[-N_last:],label="szum - yi")
ax1.legend()
ax1.set_title("Stochastyczne xi i yi")
fig.savefig("stochastyczne.png",dpi=500,pad_inches = 0)


############## autocorr
#xi
fig, ((ax1)) = plt.subplots(1, 1)
gamma_0=fun_gamma_k(x_stoch[-N_last:],0,np.average(x_stoch[-N_last:]))
sgm_dat=np.zeros(k_numb+1)
for i in range(k_numb+1):
	sgm_dat[i]=fun_gamma_k(x_stoch[-N_last:],i,np.average(x_stoch[-N_last:]))/gamma_0

ax1.plot(range(k_numb+1),sgm_dat,label="x_i")

#yi
gamma_0=fun_gamma_k(noise[-N_last:],0,np.average(noise[-N_last:]))
sgm_dat=np.zeros(k_numb+1)
for i in range(k_numb+1):
	sgm_dat[i]=fun_gamma_k(noise[-N_last:],i,np.average(noise[-N_last:]))/gamma_0

ax1.plot(range(k_numb+1),sgm_dat,label="y_i")


ax1.legend()
ax1.set_title("fun autokorelacji phi1=0.95 phi2=-0.8")
fig.savefig("fun_auto_stoch.png",dpi=500,pad_inches = 0)