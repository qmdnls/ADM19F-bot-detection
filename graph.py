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

# Check whether we have a graph saved in the file "user.graph" and load it, otherwise create from users.csv
if (os.path.exists("user.graph")):
    print("Reading graph from file...")
    G = nx.read_gpickle("user.graph")

else:
    print("Creating graph from dataset...")
    print("This can take a while.")
    G = nx.DiGraph()
    
    # Get total file length to calculate progress
    length = sum(1 for row in open('data/users.csv', 'r'))

    # Load the dataset in chunks, we'll assume it is stored in users.csv
    for chunk in load_dataset("data/users.csv"):
        G = graph_from_data(G, chunk, length)
    
    print("Writing to 'user.graph'...")
    nx.write_gpickle(G, "user.graph")

# Check whether undirected version is saved, otherwise create the file and save it
if not (os.path.exists("undirected.graph")):
    F = G.to_undirected()
    nx.write_gpickle(F, "undirected.graph")

#U = G.subgraph(users)

# Print some general stats
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
