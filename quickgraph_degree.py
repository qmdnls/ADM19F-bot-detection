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
    _, bins, patches = plt.hist(data1, bins=100, density=True, range=(0,20000), color="red", cumulative=True, histtype="step", label="Bots")
    patches[0].set_xy(patches[0].get_xy()[:-1])
    _, _, patches = plt.hist(data2, bins=bins, density=True, alpha=0.9, color="blue", label="Humans", cumulative=True, histtype="step")
    patches[0].set_xy(patches[0].get_xy()[:-1])
    plt.ylabel('Frequency (cumulative)')
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig("fig/" + filename, dpi=300)

with open('degrees.data', 'rb') as filehandle:
    # read the data as binary data stream
    degrees = pickle.load(filehandle)
    h_ind_s, h_outd_s, h_ind_p, h_outd_p, b_ind_s, b_outd_s, b_ind_p, b_outd_p = tuple(degrees)

# Plot humans against bots
plotvs(b_ind_s, h_ind_s, "Indegree of successors", "indegree_succ.pdf")
plotvs(b_outd_s, h_outd_s, "Outdegree of successors", "outdegree_succ.pdf")
plotvs(b_ind_p, h_ind_p, "Indegree of predecessors", "indegree_pre.pdf")
plotvs(b_outd_p, h_outd_p, "Outdegree of predecessors", "outdegree_pre.pdf")
