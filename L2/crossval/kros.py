import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from symfit import parameters, variables, sin, cos, Fit
import operator
import statistics

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
log="log_kr.txt"
file_l=open(log,"w")
data=np.loadtxt(file,usecols=np.arange(0,2))

fig,ax=plt.subplots()

NUM_F=5
kros_del=5

# print(data)
# data_test=data[kros_del]
# data_train=np.delete(data,(kros_del), axis=0)
# print(data)
# print(data_train)
# print(data_test)

for n in range(1,NUM_F+1):
	kros_med=0
	for n_test in range(0,len(data)):
		# print(str(n_test)+" ")
		kros_del=n_test
		data_test=data[kros_del]
		data_train=np.delete(data,(kros_del), axis=0)
		# print(len(data_train))
		# print(len(data_test))
		# print(data_test)


		plt.clf()
		file_l.write("\n\n#################\nFourier fit n="+str(n)+"\n#################\n\n")

		x, y = variables('x, y')
		w, = parameters('w')
		model_dict = {y: fourier_series(x, f=w, n=n)}
		file_l.write(str(model_dict))

		# Make step function data
		xdata = data_train[:,0]
		ydata = data_train[:,1]

		# Define a Fit object for this model and data
		fit = Fit(model_dict, x=xdata, y=ydata)
		fit_result = fit.execute()
		file_l.write(str(fit_result))

		ydata_fit=fit.model(x=xdata, **fit_result.params).y
		x_dat_linsp=np.linspace(0,1,100)

		plt.plot(xdata, ydata,marker='o',linestyle='none')
		plt.plot(xdata, ydata_fit,marker='o',linestyle='none')
		plt.plot(x_dat_linsp, fit.model(x=x_dat_linsp, **fit_result.params).y)

		plt.plot(data_test[0], fit.model(x=data_test[0], **fit_result.params).y,color="red",marker='o',linestyle='none')
		plt.plot(data_test[0], data_test[1],color="red",marker='o',linestyle='none')


		kros_med+=(data_test[1]-fit.model(x=data_test[0], **fit_result.params).y)**2


		plt.xlabel('x')
		plt.ylabel('y')
		plt.savefig("plots/FFit_"+str(n)+"_"+str(n_test)+".png")
		
		file_l.write("\n\n Fit: \n "+str(ydata_fit))
		file_l.write("\n\n Data: \n "+str(ydata))


	print("Fit" +str(n)+" - kroswalidacja średnia: "+str(kros_med/(len(data)*0.6*0.6)))

file_l.close()

