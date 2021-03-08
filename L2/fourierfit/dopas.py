import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from symfit import parameters, variables, sin, cos, Fit
import operator
import statistics
from scipy.stats import chisquare


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
    series = a0 + sum(ai * cos(i * 2*np.pi * x) + bi * sin(i * 2*np.pi * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

##########################################
file="daneszum2.txt"
log="log.txt"
file_l=open(log,"w")
data=np.loadtxt(file,usecols=np.arange(0,2))

fig,ax=plt.subplots(nrows=2, ncols=3)
# ax.plot(data[:,0],data[:,1],marker='o',linestyle='none')

chi_sq_values=[]
AIC=[]
BIC=[]
chi_ss=[]
NUM_F=5
# plt.show()

# print(data)
# data=sorted(data,key=operator.itemgetter(0))
# datax,datay=zip(*data)
# print(data)
# ax.flatten()
# print(ax)
for n,axis in zip(range(1,NUM_F+1),ax.flat):
	# plt.clf()
	file_l.write("\n\n#################\nFourier fit n="+str(n)+"\n#################\n\n")
	# ax[1]
	x, y = variables('x, y')
	w, = parameters('w')
	model_dict = {y: fourier_series(x, f=w, n=n)}
	file_l.write(str(model_dict))

	# Make step function data
	xdata = data[:,0]
	ydata = data[:,1]

	# Define a Fit object for this model and data
	fit = Fit(model_dict, x=xdata, y=ydata)
	fit_result = fit.execute()
	file_l.write(str(fit_result))

	# Plot the result
	ydata_fit=fit.model(x=xdata, **fit_result.params).y
	x_dat_linsp=np.linspace(0,1,100)#xdata.min()

	axis.plot(xdata, ydata,marker='o',linestyle='none')
	axis.plot(xdata, ydata_fit,marker='o',linestyle='none')
	axis.plot(x_dat_linsp, fit.model(x=x_dat_linsp, **fit_result.params).y)
	axis.set_xlabel('x')
	axis.set_ylabel('y')
	axis.set_title("f="+str(n))


	
	file_l.write("\n\nFit: \n "+str(ydata_fit))
	file_l.write("\n\nData: \n "+str(ydata))

	chisq=0
	for i,x in enumerate(ydata):
		chisq+=((  (ydata[i]-ydata_fit[i])/0.6  )**2)
		# chisq+=(  (ydata[i]-ydata_fit[i])**2 /ydata_fit[i]  )
		#print((ydata[i]-ydata_fit[i]/statistics.stdev(ydata))**2)
	# print(statistics.stdev(ydata))
	print("Chisq="+str(chisq))
	# print("Chisq_fit="+str(fit_result.chi_squared))
	# print("Chisq_scipy="+str(chisquare(ydata,ydata_fit)))
	print("Chisq na stop="+str(chisq/(len(xdata)-(2*n+1))))

	chiss=chisq/(len(xdata)-(2*n+1))
	chi_ss.append(chiss)
	AIC.append(chisq+(2*n+1)*2)
	BIC.append(chisq+(2*n+1)*np.log(len(xdata)))
	print("Fit" +str(n)+" fin.\n")


for n,axis in zip(range(1,7),ax.flat):
	if n==6:
		axis.axis("off")

plt.tight_layout()

plt.savefig("FFit_res.png",dpi=200,pad_inches = 0)

print("AIC:")
print(AIC)
print("BIC:")
print(BIC)

file_l.close()

print("chiss")
print(chi_ss)
print("\nAIC prob: \n")
for i in range(0,NUM_F):
	print("P"+str(i+1)+"/P_"+str(AIC.index(min(AIC))+1)+"= "+str(	np.exp(-(AIC[i]-min(AIC))/2)	))
print("\nBIC prob: \n")
for i in range(0,NUM_F):
	print("P"+str(i+1)+"/P_"+str(BIC.index(min(BIC))+1)+"= "+str(	np.exp(-(BIC[i]-min(BIC))/2)	))