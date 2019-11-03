import numpy as np
import pandas as pd
import networkx as nx
import os.path
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import IPython
import pickle
from collections import Counter

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

# List of attributes
attributes = ["label", "screen_name", "followers", "following", "name", "location", "url", "description", "protected", "listed_count", "favourites_count", "statuses_count", "created_at", "default_profile", "default_profile_image"]

# Check whether we have a graph saved in the file "user.graph" and load it, otherwise create from dataset.csv
if (os.path.exists("user.graph")):
    print("Reading graph from file...")
    G = nx.read_gpickle("user.graph")

else:
    print("Error: user.graph not found. Please run graph.py to create the graph first.")
    exit()

# Remove nodes with missing attributes (something went wrong in graph creation?) 
a = [node for node,attr in G.nodes(data=True) if "label" not in attr]
for node in a:
    G.remove_node(node)


#----------------------------------------------------------------------
#print("ind")
#ind = nx.in_degree_centrality(G)
#print("outd")
#outd = nx.out_degree_centrality(G)

#h_ind = []
#for h in humans:
#    h_ind.append(ind[h])
#
#b_ind = []
#for b in bots:
#    b_ind.append(ind[b])

#h_outd = []
#for h in humans:
#    h_outd.append(outd[h])

#b_outd = []
#for b in bots:
#    b_outd.append(outd[b])

#print("Median human in-degree centrality", statistics.median(h_ind))
#print("Median bot in-degree centrality", statistics.median(b_ind))
#print("Median human out-degree centrality", statistics.median(h_outd))
#print("Median bot in-degree centrality", statistics.median(b_outd))


#print("ev")
#ev = nx.eigenvector_centrality_numpy(G)

#h_ev = []
#for h in humans:
#    h_ev.append(ev[h])

#b_ev = []
#for b in bots:
#    b_ev.append(ev[b])

#print("Median human eigenvector centrality", statistics.median(h_ev))
#print("Median bot eigenvector centrality", statistics.median(b_ev))
#----------------------------------------------------------------------

#----------------------------------------------------------------------
## NEED STATISTICS FOR ALL 4M USERS FROM TWITTER API TO USE THIS CODE
#neighbor_followers = []
#neighbor_following = []
#neighbor_ratio = []
#for h in humans:
#    followers_neighbors = []
#    following_neighbors = []
#    friend_ratio = []
#    for s in G.successors(h):
#        followers = s['followers']
#        following = s['following']
#        ratio = followers/following
#        followers_list.append(followers)
#        following_list.append(following)
#        friend_ratio.append(ratio)
#    neighbor_followers.append(statistics.median(followers_neighbors))
#    neighbor_following.append(statistics.median(following_neighbors))
#    neighbor_ratio.append(statistics.median(friend_ratio))
#----------------------------------------------------------------------

df = pd.read_csv("users.csv", encoding="utf8")
df.set_index("user_id")

humans = [node for node,attr in G.nodes(data=True) if attr['label'] == 'human']
bots = [node for node,attr in G.nodes(data=True) if attr['label'] == 'bot']

following = nx.get_node_attributes(G, 'following')
followers = nx.get_node_attributes(G, 'followers')

h_rep_s = []
h_rep_p = []
for h in humans:
    rep_succ = []
    rep_pre = []
    for s in G.successors(h):
        a = followers[s]
        b = following[s]
        if (a == 0 and b == 0):
            rep_succ.append(0)
        else:
            rep_succ.append(a/(a+b))
    for p in G.predecessors(h):
        a = followers[p]
        b = following[p]
        if (a == 0 and b == 0):
            rep_pre.append(0)
        else:
            rep_pre.append(a/(a+b))

    if (len(rep_succ) > 0):
        h_rep_s.append(statistics.median(rep_succ))
    else:
        h_rep_s.append(0) 

    if (len(rep_pre) > 0):
        h_rep_p.append(statistics.median(rep_pre))
    else:
        h_rep_p.append(0)
    
b_rep_s = []
b_rep_p = []
for b in bots:
    rep_succ = []
    rep_pre = []
    for s in G.successors(b):
        a = followers[s]
        c = following[s]
        if (a == 0 and c == 0):
            rep_succ.append(0)
        else:
            rep_succ.append(a/(a+c))
    for p in G.predecessors(b):
        a = followers[p]
        c = following[p]
        if (a == 0 and c == 0):
            rep_pre.append(0)
        else:
            rep_pre.append(a/(a+c))

    if (len(rep_succ) > 0):
        b_rep_s.append(statistics.median(rep_succ))
    else:
        b_rep_s.append(0) 

    if (len(rep_pre) > 0):
        b_rep_p.append(statistics.median(rep_pre))
    else:
        b_rep_p.append(0)
 
# Print some general stats
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

# Plot humans against bots
plotvs(b_rep_s, h_rep_s, "Reputation", "reputation_succ.png")
plotvs(b_rep_p, h_rep_p, "Reputation", "reputation_pre.png")

rep = [h_rep_s, h_rep_p, b_rep_s, b_rep_p]

with open('reputation.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(rep, filehandle)
