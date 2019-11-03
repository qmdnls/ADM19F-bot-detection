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
    plt.hist([data1, data2], weights=[weights1, weights2], bins=20, density=False, log=True, color=["red", "blue"], cumulative=False, label=["Bots","Human"])
    plt.ylabel("Frequency")
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig("fig/" + filename, dpi=300)

with open('centrality.data', 'rb') as filehandle:
    # read the data as binary data stream
    centrality = pickle.load(filehandle)
    h_cent, b_cent = tuple(centrality)

# Plot humans against bots
plotvs(b_cent, h_cent, "Eigenvector centrality", "centrality.png")
