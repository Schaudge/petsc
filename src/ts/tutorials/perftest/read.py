import utils as ut
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#def table(machine):
	# get data
	#sm = range(1000, 100001, 100)
#sm = 1000000
	
operations = ["VecDot", "VecAXPY", "VecSet", "VecCopy", "VecCUDACopyTo"]
counts = [1, 3, 3, 1, 1]

file_in = "../rsperf_{}_{}.log"
file_out = "timings.log"

msizes = [100000]
print(" {} \n".format(msizes[0]))
print(" {} \n".format(len(msizes)))

file = open(file_out, "w")
file.write("ConfigID Msize VecDot VecAXPY VecSet VecCopy VecCUDACopyTo\n")
file.close()

for ic in range(3): 
	for im in range(len(msizes)):
		file = open(file_out, "a+")
		
		file.write(" {} {} ".format(ic+1, msizes[im]))

		for i in range(len(operations)):
			sm_data = []
			sm_data.append(float(ut.get_time(file_in.format(msizes[im],ic+1), operations[i], counts[i])))
			sm_data = np.array(sm_data)
			file.write(" {} ".format(sm_data[0]))

		file.write(" \n ")
	
file.close()
