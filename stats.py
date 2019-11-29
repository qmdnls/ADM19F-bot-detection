import numpy as np
import pandas as pd
import networkx as nx
import os.path
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import IPython
import pickle
import time
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

# Check whether we have a graph saved in the file "user.graph" and load it
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

#print("Loading dataset...")
#df = pd.read_csv("data/users.csv", encoding="utf8")
#df.set_index("user_id")

humans = [node for node,attr in G.nodes(data=True) if attr['label'] == 'human']
bots = [node for node,attr in G.nodes(data=True) if attr['label'] == 'bot']
users = humans + bots

print("Retrieve node attributes from the graph...")
following = nx.get_node_attributes(G, 'following')
followers = nx.get_node_attributes(G, 'followers')
created_at = nx.get_node_attributes(G, 'created_at')
favorites_count = nx.get_node_attributes(G, 'favourites_count')
statuses_count = nx.get_node_attributes(G, 'statuses_count')
listed_count = nx.get_node_attributes(G, 'listed_count')
default_profile = nx.get_node_attributes(G, 'default_profile')
default_profile_image = nx.get_node_attributes(G, 'default_profile_image')
default_profile = {k: int(v) for k,v in default_profile.items()}
default_profile_image = {k: int(v) for k,v in default_profile_image.items()}

print("Get node neighboor attributes...")

# Create a dict for every neighborhood feature so we can add it to the dataframe later, suffix s for successors, p for predecessors in the graph
indegree_predecessors = {}
indegree_successors = {}
outdegree_predecessors = {}
outdegree_successors = {}
reputation_predecessors = {}
reputation_successors = {}
favorites_predecessors = {}
favorites_successors = {}
status_predecessors = {}
status_successors = {}
listed_predecessors = {}
listed_successors = {}
age_predecessors = {}
age_successors = {}
default_predecessors = {}
default_successors = {}
default_image_predecessors = {}
default_image_successors = {}
# Create dicts for graph features
ego_edges = {}
ego_nodes = {}
ego_density = {}
ego_reciprocity = {}
ego_assortativity = {}

# For each user, get the successor and predecessor features as a list so we can calculate the median and use that
time = time.time()
length = len(users)
for idx, user in enumerate(users):
    # Progress
    print("User", idx, "of", length, "(" + str(round(100*idx/length)) + "%)", end="\r", flush=True)
    # Temp vars for one user iteration
    ind = []
    outd = []
    rep = []
    fav = []
    status = []
    listed = []
    age = []
    default = []
    default_image = []
    features = [ind, outd, rep, fav, status, listed, age, default, default_image]
    for successor in G.successors(user):
        # Retrieve the features
        indegree = followers[successor]
        outdegree = following[successor]
        reputation = 0
        if (indegree > 0 or outdegree > 0):
            reputation = indegree / (indegree + outdegree)
        age_ = (time - created_at[successor] / 10**9) / 60 / 60 / 24 # Convert nanoseconds to seconds to subtract and obtain the age, then convert the age from seconds to days
        # Append to successor list
        ind.append(indegree)
        outd.append(outdegree)
        rep.append(reputation)
        fav.append(favorites_count[successor])
        status.append(statuses_count[successor])
        listed.append(listed_count[successor])
        age.append(age_)
        default.append(default_profile[successor])
        default_image.append(default_profile_image[successor])
    for feature in features:
        if (len(feature) == 0):
            feature.append(0)
    indegree_successors[user] = statistics.median(ind)    
    outdegree_successors[user] = statistics.median(outd)
    reputation_successors[user] = statistics.median(rep)
    favorites_successors[user] = statistics.median(fav)
    status_successors[user] = statistics.median(status)
    listed_successors[user] = statistics.median(listed)
    age_successors[user] = statistics.median(age)
    default_successors[user] = statistics.median(default)
    default_image_successors[user] = statistics.median(default_image)
    # Repeat the same thing for predecessors
    ind = []
    outd = []
    rep = []
    fav = []
    status = []
    listed = []
    age = []
    default = []
    default_image = []
    features = [ind, outd, rep, fav, status, listed, age, default, default_image]
    for predecessor in G.predecessors(user):
        # Retrieve the features
        indegree = followers[predecessor]
        outdegree = following[predecessor]
        reputation = 0
        if (indegree > 0 or outdegree > 0):
            reputation = indegree / (indegree + outdegree)
        age_ = (time - created_at[predecessor] / 10**9) / 60 / 60 / 24 # Convert nanoseconds to seconds to subtract and obtain the age, then convert the age from seconds to days
        # Append to successor list
        ind.append(indegree)
        outd.append(outdegree)
        rep.append(reputation)
        fav.append(favorites_count[predecessor])
        status.append(statuses_count[predecessor])
        listed.append(listed_count[predecessor])
        age.append(age_)
        default.append(default_profile[predecessor])
        default_image.append(default_profile_image[predecessor])
    for feature in features:
        if (len(feature) == 0):
            feature.append(0)
    # Set neighborhood features
    indegree_predecessors[user] = statistics.median(ind) 
    outdegree_predecessors[user] = statistics.median(outd)
    reputation_predecessors[user] = statistics.median(rep)
    favorites_predecessors[user] = statistics.median(fav)
    status_predecessors[user] = statistics.median(status)
    listed_predecessors[user] = statistics.median(listed)
    age_predecessors[user] = statistics.median(age)
    default_predecessors[user] = statistics.median(default)
    default_image_predecessors[user] = statistics.median(default_image)
    # Set graph features
    ego = nx.ego_graph(G, user)
    ego_nodes[user] = ego.number_of_nodes()
    ego_edges[user] = ego.number_of_edges()
    ego_density[user] = nx.density(ego)
    try:
        ego_reciprocity[user] = nx.reciprocity(ego)
    except:
        ego_reciprocity[user] = 0
    #ego_assortativity[user] = nx.attribute_assortativity_coefficient(ego, "followers")

data = [indegree_predecessors, indegree_successors, outdegree_predecessors, outdegree_successors, reputation_predecessors, reputation_successors, favorites_predecessors, favorites_successors, status_predecessors, status_successors, listed_predecessors, listed_successors, age_predecessors, age_successors, default_predecessors, default_successors, default_image_predecessors, default_image_successors, ego_nodes, ego_edges, ego_density, ego_reciprocity, ego_assortativity]

with open('data/neighbor.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(data, filehandle)

# Read the data like this:
# with open('data/neighbor.data', 'rb') as filehandle:
#     data = pickle.load(filehandle)
#     indegree_predecessors_data, indegree_successors_data, outdegree_predecessors_data, outdegree_successors_data, reputation_predecessors_data, reputation_successors_data, favorites_predecessors_data, favorites_successors_data, status_predecessors_data, status_successors_data, listed_predecessors_data, listed_successors_data, age_predecessors_data, age_successors_data, default_predecessors_data, default_successors_data, default_image_predecessors_data, default_image_successors_data, ego_nodes_data, ego_edges_data, ego_density, ego_reciprocity, ego_assortativity = tuple(data)

del G

print("Loading dataset...")
df = pd.read_csv("data/users.csv", encoding="utf8")
df.set_index("user_id")

# Set column names
features = ("indegree_predecessors", "indegree_successors", "outdegree_predecessors", "outdegree_successors", "reputation_predecessors", "reputation_successors", "favorites_predecessors", "favorites_successors", "status_predecessors", "status_successors", "listed_predecessors", "listed_successors", "age_predecessors", "age_successors", "default_predecessors", "default_successors", "default_image_predecessors", "default_image_successors", "ego_edges", "ego_nodes", "ego_density", "ego_reciprocity", "ego_assortativity")
d = dict(zip(features, tuple(data)))

print("Merge neighborhood features into training dataset...")
for name, feature in d.items():
    df[name] = df['user_id'].map(feature)

print("Write dataset to file...")
df.head()
df.to_csv("data/users_neighborhood.csv", index=None, header=True)

print("Done.")
