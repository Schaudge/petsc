from run_ex2 import run_test
import numpy as np


nsolves = np.arange(1, 33, 1)
timers = np.zeros(nsolves.size)

for i in range(0, len(nsolves)):
    timers[i] = run_test(nsolves_team=nsolves[i])

print(timers)