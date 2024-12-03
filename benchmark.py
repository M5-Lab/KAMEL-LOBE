# Python script for benchmarking memory and time usage of the KAMEL_LOBE implemintation in KAMEL_LOBE.py
import numpy as np
import cProfile
from KAMEL_LOBE import KAMEL_LOBE as KL
import matplotlib.pyplot as plt

test_files = ['RDF_Sample_1.txt', 'RDF_Sample_2.txt']
r_vals = []
RDF_vals = []


def test_function(r, RDF):
    return KL(r, RDF)

# Loop for loading in data
# Seperate so we can test preformance of KAMEL_LOBE separately from loading in data
for test_file in test_files:
    data = np.loadtxt(test_file)
    r = data[:,0]
    RDF = data[:,1]
    r_vals.append(r)
    RDF_vals.append(RDF)

# Run the profiler
profiler = cProfile.Profile()
profiler.enable()

vals = test_function(r_vals[0], RDF_vals[0])

profiler.disable()
#profiler.print_stats(sort='cumulative')

# plot
plt.plot(r_vals[0], RDF_vals[0], label='RDF')
plt.plot(vals[0], vals[1], label='KAMEL_LOBE')
plt.legend()
plt.show()









