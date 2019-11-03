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


#U = G.subgraph(users)

#----------------------------------------------------------------------
#indegrees = sorted(G.in_degree, key=lambda tup: tup[1], reverse=True)
#outdegrees = sorted(G.out_degree, key=lambda tup: tup[1], reverse=True)
#print(nx.info(U))
#humans = [attr['statuses_count'] for node,attr in U.nodes(data=True) if attr['label'] == 'human']
#bots = [attr['statuses_count'] for node,attr in U.nodes(data=True) if attr['label'] == 'bot']
#----------------------------------------------------------------------

#humans = [node for node,attr in U.nodes(data=True) if attr['label'] == 'human']
#bots = [node for node,attr in U.nodes(data=True) if attr['label'] == 'bot']

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

#print(df[df['user_id'] == 755095269730095104])
#print([s for s in G.successors(755095269730095104)])

humans = [node for node,attr in G.nodes(data=True) if attr['label'] == 'human']
bots = [node for node,attr in G.nodes(data=True) if attr['label'] == 'bot']

following = nx.get_node_attributes(G, 'following')
followers = nx.get_node_attributes(G, 'followers')

h_ind_s = []
h_outd_s = []
h_ind_p = []
h_outd_p = []
for h in humans:
    indegree_succ = []
    outdegree_succ = []
    indegree_pre = []
    outdegree_pre = []
    for s in G.successors(h):
        indegree_succ.append(followers[s])
        outdegree_succ.append(following[s])
    for p in G.predecessors(h):
        indegree_pre.append(followers[p])
        outdegree_pre.append(following[p])

    if (len(indegree_succ) > 0):
        h_ind_s.append(statistics.median(indegree_succ))
    else:
        h_ind_s.append(0) 
    if (len(outdegree_succ) > 0):
        h_outd_s.append(statistics.median(outdegree_succ))
    else:
        h_outd_s.append(0)

    if (len(indegree_pre) > 0):
        h_ind_p.append(statistics.median(indegree_pre))
    else:
        h_ind_p.append(0)
    if (len(outdegree_pre) > 0):
        h_outd_p.append(statistics.median(outdegree_pre))
    else:
        h_outd_p.append(0)
    
print("Median mean neighbor in-degree of successors human",statistics.median(h_ind_s))
print("Median mean neighbor out-degree of successors human",statistics.median(h_outd_s))
print("Median mean neighbor in-degree of predecessors human",statistics.median(h_ind_p))
print("Median mean neighbor out-degree of predecessors human",statistics.median(h_outd_p))

# Make some graphs
#plot(h_ind_s, "Indegree", "h_indegree_succ.png")
#plot(h_outd_s, "Outdegree", "h_outdegree_succ.png")
#plot(h_ind_p, "Indegree", "h_indegree_pre.png")
#plot(h_outd_p, "Outdegree", "h_outdegree_pre.png")

b_ind_s = []
b_outd_s = []
b_ind_p = []
b_outd_p = []
for b in bots:
    indegree_succ = []
    outdegree_succ = []
    indegree_pre = []
    outdegree_pre = []
    for s in G.successors(b):
        indegree_succ.append(followers[s])
        outdegree_succ.append(following[s])
    for p in G.predecessors(b):
        indegree_pre.append(followers[p])
        outdegree_pre.append(following[p])

    if (len(indegree_succ) > 0):
        b_ind_s.append(statistics.mean(indegree_succ))
    else:
        b_ind_s.append(0)
    if (len(outdegree_succ) > 0):
        b_outd_s.append(statistics.mean(outdegree_succ))
    else:
        b_outd_s.append(0)

    if (len(indegree_pre) > 0):
        b_ind_p.append(statistics.mean(indegree_pre))
    else:
        b_ind_p.append(0)
    if (len(outdegree_pre) > 0):
        b_outd_p.append(statistics.mean(outdegree_pre))
    else:
        b_outd_p.append(0)
    
print("Median mean neighbor in-degree of successors bots",statistics.median(b_ind_s))
print("Median mean neighbor out-degree of successors bots",statistics.median(b_outd_s))
print("Median mean neighbor in-degree of predecessors bots",statistics.median(b_ind_p))
print("Median mean neighbor out-degree of predecessors bots",statistics.median(b_outd_p))

# Make some graphs
#plot(b_ind_s, "Indegree", "b_indegree_succ.png")
#plot(b_outd_s, "Outdegree", "b_outdegree_succ.png")
#plot(b_ind_p, "Indegree", "b_indegree_pre.png")
#plot(b_outd_p, "Outdegree", "b_outdegree_pre.png")

# Print some general stats
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

human_in = [i[1] for i in list(G.in_degree(humans))]
human_out = [i[1] for i in list(G.out_degree(humans))]
bots_in = [i[1] for i in list(G.in_degree(bots))]
bots_out = [i[1] for i in list(G.out_degree(bots))]

print("Human: mean in-degree", statistics.mean(human_in))
print("Bots: mean in-degree", statistics.mean(bots_in))
print("Human: mean out-degree", statistics.mean(human_out))
print("Bots: mean out-degree", statistics.mean(bots_out))

print("Human: average in-degree", statistics.median(human_in))
print("Bots: average in-degree", statistics.median(bots_in))
print("Human: average out-degree", statistics.median(human_out))
print("Bots: average out-degree", statistics.median(bots_out))

# Plot humans against bots
plotvs(b_ind_s, h_ind_s, "Indegree", "indegree_succ.png")
plotvs(b_outd_s, h_outd_s, "Outdegree", "outdegree_succ.png")
plotvs(b_ind_p, h_ind_p, "Indegree", "indegree_pre.png")
plotvs(b_outd_p, h_outd_p, "Outdegree", "outdegree_pre.png")

degrees = [h_ind_s, h_outd_s, h_ind_p, h_outd_p, b_ind_s, b_outd_s, b_ind_p, b_outd_p]

with open('listfile.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(degrees, filehandle)
