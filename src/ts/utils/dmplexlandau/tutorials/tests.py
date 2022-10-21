from run_ex2 import run_test
import numpy as np

timers = np.array([])
timers = np.append(timers, run_test())
timers = np.append(timers, run_test(ortho_strategy=1))
timers = np.append(timers, run_test(temp_data_strategy=1))
timers = np.append(timers, run_test(ortho_strategy=1,temp_data_strategy=1))
timers = np.append(timers, run_test(shared_level=1))
timers = np.append(timers, run_test(shared_level=1,ortho_strategy=1))
timers = np.append(timers, run_test(shared_level=1,temp_data_strategy=1))
timers = np.append(timers, run_test(shared_level=1,ortho_strategy=1,temp_data_strategy=1))

print(timers)