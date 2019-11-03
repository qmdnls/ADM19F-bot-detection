import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Histogram plotting function
def plot(data, xlabel, filename):
    plt.figure()
    plt.hist(data, bins=40, log=True, density=False, range=(0, 500000), cumulative=True)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.savefig("fig/" + filename)

def plotvs(data1, data2, xlabel, filename):
    plt.figure()
    # Norm
    weights1 = np.ones_like(data1)/float(len(data1))
    weights2 = np.ones_like(data2)/float(len(data2))
    plt.hist([data1, data2], weights=[weights1, weights2], bins=20, density=False, range=(0,0.01), log=True, color=["red", "blue"], cumulative=False, label=["Bots","Human"])
    plt.ylabel("Frequency")
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig("fig/" + filename, dpi=300)

with open('outdegreescatter.data', 'rb') as filehandle:
    # read the data as binary data stream
    data = pickle.load(filehandle)
    h_in_values, h_in_hist, b_in_values, b_in_hist = tuple(data)

# Plot humans against bots
plt.figure()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, 10000)
plt.ylim(1, 1000)
plt.scatter(h_in_values, h_in_hist, color="blue", s=3, label="Human")
plt.scatter(b_in_values, b_in_hist,  color="red", s=3, label="Bots")
plt.xlabel('Outdegree')
plt.ylabel('Count')
plt.legend()
plt.savefig('fig/outdegrees.png', dpi=300)
