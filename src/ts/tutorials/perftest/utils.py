import re

# get operation time from -log_view
#     operation: line starts with operation name
#     count: which ocurrance of operation
def get_time(file_name, operation, count):
	file = open(file_name, "r")
	for line in file:
		if operation in line:
			if count == 1:
				return float(re.findall("\d+\.\d+[eE][+-]\d+", line)[0])
			else:
				count -= 1
	return float('nan') # CUDA error

# get flop rate from -log_view file
#     operation: line starts with operation name
#     CPU: True if CPU floprate, False if GPU floprate
#     count: which ocurrance of operation
def get_floprate(file_name, operation, CPU, count):
	file = open(file_name, "r")
	for line in file:
		if operation in line:
			if count == 1:
				line = re.findall("\d+", line)
				if CPU:
					return line[-11] # CPU
				else: 
					return line[-10] # GPU
			else:
				count -= 1
				continue
	return float('nan') # CUDA error

# calculate rate in 8 Mbytes/seconds
def calc_rate(vec_length, time):

	# x*10-6(Mbyes)/time
	return (vec_length*(10**(-6)))/(time*1.0)

