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

def fun_density(x,mu,sgm):
	S=0.2
	y=1

	for i in x:
		y=y*(S*(1/(np.sqrt(2*np.pi)*sgm))*np.exp(-((i-mu)**2)/(2*(sgm**2)))+(1-S)/18) #*fun_p_aprioi(mu,sgm)

	return y

def fun_p_aprioi(mu,sgm):
	fun=-4.15+3.6*mu-0.6*mu**2+4.8*sgm-12*sgm**2
	return fun



#contains minor bugs

file="obserwacje.dat"
file_l=open("log_s.txt","w")
file_l_metro=open("log_metro.txt","w")
x_data=np.loadtxt(file,usecols=np.arange(0,1))

file_l.write("\n\nData list size: \n "+str(len(x_data)))

step_mu=0.1
step_sgm=0.01

mu_data=np.arange(2,4+step_mu,step_mu)
sgm_data=np.arange(0.1,0.5+step_sgm,step_sgm)


popost=np.zeros((mu_data.shape[0],sgm_data.shape[0]))
norm=0
apop=np.zeros((mu_data.shape[0],sgm_data.shape[0]))

file_l.write("\n\nMu list shape: \n "+str(mu_data.shape))
file_l.write("\n\nMu data: \n "+str(mu_data))
file_l.write("\n\nSgm list shape: \n "+str(sgm_data.shape))
file_l.write("\n\nSgm data: \n "+str(sgm_data))

file_l.write("\n\nPosteriori data shape: \n "+str(popost.shape))



for i,im in enumerate(mu_data):
	for j,js in enumerate(sgm_data):
		popost[i,j]=fun_density(x_data,im,js)*fun_p_aprioi(im,js)
		apop[i,j]=fun_p_aprioi(im,js)
		# norm+=fun_p_aprioi(im,js)*step_sgm*step_mu
		

norm=popost.sum()
# file_l.write("\n\nnorma: "+str(norm))

# file_l.write("\n\nData: \n "+str(popost))

popost=popost/norm

# file_l.write("\n\nData after normalisation: \n "+str(popost))

mu_avg=0
mu_dev=0

sgm_avg=0
sgm_dev=0

suma=0

for i,im in enumerate(mu_data):
	for j,js in enumerate(sgm_data):
		suma+=popost[i,j]

file_l.write("\n\nSuma prawodpodobienstwa po poposteriori: "+str(suma))

##################### - z tym wartości mają jakikolwiek sens
# popost=popost/suma

im = plt.imshow(apop.T, cmap=plt.cm.magma, aspect='auto',extent=(2, 4,0.1,0.5)) #)  

plt.colorbar(im)  
plt.xlabel('mu')
plt.ylabel('sgm')
plt.title("aposter")
plt.savefig("2D_apoposter.png",dpi=200,pad_inches = 0)
plt.close()

img = plt.imshow(popost.T, cmap=plt.cm.magma, aspect='auto',extent=(2, 4,0.1,0.5)) #)  

plt.colorbar(img)  
plt.xlabel('mu')
plt.ylabel('sgm')
plt.title("pposter")
plt.savefig("2D_Pposter.png",dpi=200,pad_inches = 0)
plt.close()



for i,im in enumerate(mu_data):
	for j,js in enumerate(sgm_data):
		mu_avg+=popost[i,j]*im
		sgm_avg+=popost[i,j]*js

for i,im in enumerate(mu_data):
	for j,js in enumerate(sgm_data):
		mu_dev+=popost[i,j]*(im-mu_avg)**2
		sgm_dev+=popost[i,j]*(js-sgm_avg)**2

mu_dev=np.sqrt(mu_dev)
sgm_dev=np.sqrt(sgm_dev)

file_l.write("\n\nMu_avg: \n "+str(mu_avg))
file_l.write("\n\nSgm_avg: \n "+str(sgm_avg))
file_l.write("\n\nMu_dev: \n "+str(mu_dev))
file_l.write("\n\nSgm_dev: \n "+str(sgm_dev))


plt.plot(sgm_data,popost.sum(axis=0)*step_mu)
plt.xlabel('sgm')
plt.ylabel('mu')
plt.savefig("2D_sgm_proj.png",dpi=200,pad_inches = 0)
plt.close()
plt.xlabel('mu')
plt.ylabel('sgm')
plt.plot(mu_data, popost.sum(axis=1))
plt.savefig("2D_mup_proj.png",dpi=200,pad_inches = 0)
plt.close()
##################################################################################################
########################################### Metropolis ###########################################
##################################################################################################

rng = default_rng()
###### parameters ######
N=500
n_steps=100
delta_mu=0.05
delta_sgm=0.005
file_l_metro.write("Parameters: "+"N: "+str(N)+" n: "+str(n_steps)+" delta_mu: "+str(delta_mu)+" delta_sgm: "+str(delta_sgm))
########################
mi_gen=np.zeros(N)
sgm_gen=np.zeros(N)



# print(mi_gen)

start_time = time.time()

for i in np.arange(N):
	mu_start=np.random.uniform(2,4)
	sgm_start=np.random.uniform(0.1,0.5)
	# print(mu_start)
	for j in np.arange(n_steps):
		u=np.random.uniform(0,1)
		mu_new=mu_start+np.random.normal(0,delta_mu)
		sgm_new=sgm_start+np.random.normal(0,delta_sgm)
		if (fun_p_aprioi(mu_new,sgm_new)*fun_density(x_data,mu_new,sgm_new))>(u*fun_density(x_data,mu_start,sgm_start)*fun_p_aprioi(mu_start,sgm_start)):
			if (mu_new<4)&(mu_new>2)&(sgm_new<0.5)&(sgm_new>0.1):
				mu_start=mu_new
				sgm_start=sgm_new

	mi_gen[i]=mu_start
	sgm_gen[i]=sgm_start
	if(i%(N/50)==0):
		print(str(int(100*i/N))+"% \t Elapsed time: "+str(int(time.time()-start_time)) +"s")
	# print(mu_start)

# print(mi_gen)	

file_l_metro.write("\n\nMu_avg: \n "+str(np.average(mi_gen)))
file_l_metro.write("\n\nSgm_avg: \n "+str(np.average(sgm_gen)))	

file_l_metro.write("\n\nMu_dev: \n "+str(np.std(mi_gen)))
file_l_metro.write("\n\nSgm_dev: \n "+str(np.std(sgm_gen)))

file_l_metro.write("\n\nData mu : \n "+str(mi_gen))
file_l_metro.write("\n\nData sgm: \n "+str(sgm_gen))

fig = plt.figure()

plt.hist2d(mi_gen, sgm_gen,cmap=plt.cm.magma,bins=int(np.sqrt(np.count_nonzero(mi_gen)))) 
plt.title("Simple 2D Histogram") 
plt.colorbar()  
# show plot 
plt.savefig("3Dhist.png",dpi=200,pad_inches = 0)
print(int(np.sqrt(np.count_nonzero(mi_gen))))