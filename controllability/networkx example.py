import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load in the edge list
# You can replace this file with another edge list in the same folder if you want to try something else
edge_list_df = pd.read_csv('./feeding network 10 percent volume.csv')

# Make the graph from the edge list
# This is a directed graph using origin_abbrev -> target_abbrev and projection_volume for the weights
G = nx.from_pandas_edgelist(edge_list_df, 'origin_abbrev', 'target_abbrev', edge_attr = 'projection_volume', create_using=nx.DiGraph())
nodes = G.nodes()


# Print the output of the algorithm
# Replace with closeness_centrality with whatever you want to use
print(nx.closeness_centrality(G, distance='projection_volume'))