import pandas as pd
import networkx as nx
import ast
import os.path

# Given a dataset of users with UIDs, and lists of following/followers, create the social graph
def graph_from_data(dataset):
    # Load the dataset
    df = pd.read_csv(dataset, encoding='utf8', engine='python')
    df_length = len(df)

    # Create (directed) user graph
    G = nx.DiGraph()

    # Iterate over dataset and create node for each user in the dataset as well as each user followed by or following a user in the dataset
    for idx, user in df.iterrows():
        # Print progress
        print(idx, "of", df_length, "rows processed", "(" + str(round(100*idx/df_length)) + "%)", end="\r", flush=True)

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
    
    print("\nDone.")

    # Return the generated graph
    return G

# Check whether we have a graph saved in the file "user.graph" and load it, otherwise create from dataset.csv
if (os.path.exists("user.graph")):
    print("Reading graph from file...")
    G = nx.read_gpickle("user.graph")
else:
    print("Creating graph from dataset...")
    G = graph_from_data("dataset.csv")
    print("Writing to 'user.graph'...")
    nx.write_gpickle(G, "user.graph")

# Print some general stats
print(G.number_of_nodes(), G.number_of_edges())
