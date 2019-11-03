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

with open('coeff.data', 'rb') as filehandle:
    # read the data as binary data stream
    coeff = pickle.load(filehandle)
    h_coeffs, b_coeffs = tuple(coeff)

#h_coeffs = [c*100 for c in h_coeffs]
#b_coeffs = [c*100 for c in b_coeffs]

# Plot humans against bots
plotvs(b_coeffs, h_coeffs, "Clustering coefficient", "coefficient.pdf")
