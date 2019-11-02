import numpy as np
import pandas as pd
import networkx as nx
import ast
import os.path
import statistics
    
# Load the dataset in chunks, returns list of chunks that we can iterate over
def load_dataset(filename):
    return pd.read_csv(filename, encoding='utf8', engine='python', chunksize=500000)

# Given a dataset of users with UIDs, and lists of following/followers as well as a dictionary with all IDs of labelled users, create the social graph
def graph_from_data(G, chunk, length):
    # Iterate over dataset and create node for each user in the dataset as well as each user followed by or following a user in the dataset
    for idx, user in chunk.iterrows(): 
        # Print progress
        print(idx, "of", length, "rows processed", "(" + str(round(100*idx/length)) + "%)", end="\r", flush=True)
        
        if pd.isnull(user['label']):
            continue

        # Get the user id
        uid = user['user_id']
   
        # Add node for user from dataset in the graph (if it doesn't already exist)
        G.add_node(uid)

        # Get follower and following lists, each is saved as a string representation of a list. This list contains the list we want at the 0-index. We extract by evaluating the string as a list, then getting the 0-th element.
        followers = ast.literal_eval(user['followers_list'])[0]
        following = ast.literal_eval(user['following_list'])[0]

        # Add a node for each user following/followed (if it doesn't already exist) as well as a directed edge
        for follower_id in followers:
            G.add_node(follower_id)
            G.add_edge(follower_id, uid)
        for following_id in following:
            G.add_node(following_id)
            G.add_edge(uid, following_id)
    
    attributes = chunk.set_index('user_id').T.to_dict()
    nx.set_node_attributes(G, attributes)

    # Return the generated graph and a list of users in the dataset
    return G
    
# List of attributes
attributes = ["label", "screen_name", "followers", "following", "name", "location", "url", "description", "protected", "listed_count", "favourites_count", "statuses_count", "created_at", "default_profile", "default_profile_image"]

# Check whether we have a graph saved in the file "user.graph" and load it, otherwise create from dataset.csv
if (os.path.exists("user.graph")):
    print("Reading graph from file...")
    G = nx.read_gpickle("user.graph")

else:
    print("Creating graph from dataset...")
    G = nx.DiGraph()
    
    # Get total file length to calculate progress
    length = sum(1 for row in open('users.csv', 'r'))

    # Load the dataset in chunks, we'll assume it is stored in users.csv
    for chunk in load_dataset("users.csv"):
        G = graph_from_data(G, chunk, length)
    
    print("Writing to 'user.graph'...")
    nx.write_gpickle(G, "user.graph")

# Check whether undirected version is saved, otherwise create the file and save it
if not (os.path.exists("undirected.graph")):
    F = G.to_undirected()
    nx.write_gpickle(F, "undirected.graph")

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

#ind_s = []
#outd_s = []
#ind_p = []
#outd_p = []
#for h in humans:
#    indegree_succ = []
#    outdegree_succ = []
#    indegree_pre = []
#    outdegree_pre = []
#    for s in G.successors(h):
#        indegree_succ.append(G.in_degree(s))
#        outdegree_succ.append(G.out_degree(s))
#    for p in G.predecessors(h):
#        indegree_pre.append(G.in_degree(p))
#        outdegree_pre.append(G.out_degree(p))
#
#    if (len(indegree_succ) > 0):
#        ind_s.append(statistics.mean(indegree_succ))
#    else:
#        ind_s.append(0)
#    if (len(outdegree_succ) > 0):
#        outd_s.append(statistics.mean(outdegree_succ))
#    else:
#        outd_s.append(0)
#
#    if (len(indegree_pre) > 0):
#        ind_p.append(statistics.mean(indegree_pre))
#    else:
#        ind_p.append(0)
#    if (len(outdegree_pre) > 0):
#        outd_p.append(statistics.mean(outdegree_pre))
#    else:
#        outd_p.append(0)
    
#print("Median mean neighbor in-degree of successors human",statistics.median(ind_s))
#print("Median mean neighbor out-degree of successors human",statistics.median(outd_s))
#print("Median mean neighbor in-degree of predecessors human",statistics.median(ind_p))
#print("Median mean neighbor out-degree of successors human",statistics.median(outd_p))

#ind_s = []
#outd_s = []
#ind_p = []
#outd_p = []
#for b in bots:
#    indegree_succ = []
#    outdegree_succ = []
#    indegree_pre = []
#    outdegree_pre = []
#    for s in G.successors(b):
#        indegree_succ.append(G.in_degree(s))
#        outdegree_succ.append(G.out_degree(s))
#    for p in G.predecessors(b):
#        indegree_pre.append(G.in_degree(p))
#        outdegree_pre.append(G.out_degree(p))
#
#    if (len(indegree_succ) > 0):
#        ind_s.append(statistics.mean(indegree_succ))
#    else:
#        ind_s.append(0)
#    if (len(outdegree_succ) > 0):
#        outd_s.append(statistics.mean(outdegree_succ))
#    else:
#        outd_s.append(0)
#
#    if (len(indegree_pre) > 0):
#        ind_p.append(statistics.mean(indegree_pre))
#    else:
#        ind_p.append(0)
#    if (len(outdegree_pre) > 0):
#        outd_p.append(statistics.mean(outdegree_pre))
#    else:
#        outd_p.append(0)
    
#print("Median mean neighbor in-degree of successors bots",statistics.median(ind_s))
#print("Median mean neighbor out-degree of successors bots",statistics.median(outd_s))
#print("Median mean neighbor in-degree of predecessors bots",statistics.median(ind_p))
#print("Median mean neighbor out-degree of successors bots",statistics.median(outd_p))

# Print some general stats
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges(), "Triangles:", "???")
