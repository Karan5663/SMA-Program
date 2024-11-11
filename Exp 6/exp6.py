import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the Facebook dataset (pseudo_facebook.csv)
df = pd.read_csv("C:/Users/Karan/OneDrive/Desktop/sma/SMA_PROGRAMS/Exp 6/pseudo_facebook.csv")
print(df.head())  # Check the first few rows of the data

# Load the network as a graph based on edge list
# 'age' and 'dob_year' are attributes in this dataset; modify based on your dataset
fb_graph = nx.from_pandas_edgelist(df, source="age", target="dob_year")
print("Nodes in the graph:", fb_graph.nodes())
print("Edges in the graph:", fb_graph.edges())

# Visualization of the graph
plt.figure(figsize=(10,10))
nx.draw(fb_graph, with_labels=True, node_size=20, font_size=8, node_color="skyblue", edge_color="gray")
plt.title("Facebook Network Visualization")
plt.show()

# --- Centrality Measures ---
# 1. Degree Centrality
degree_centrality = nx.degree_centrality(fb_graph)
print("Top 5 nodes by degree centrality:", sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

# 2. Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(fb_graph, normalized=True)
print("Top 5 nodes by betweenness centrality:", sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

# 3. Closeness Centrality
closeness_centrality = nx.closeness_centrality(fb_graph)
print("Top 5 nodes by closeness centrality:", sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

# 4. Eigenvector Centrality
eigenvector_centrality = nx.eigenvector_centrality(fb_graph)
print("Top 5 nodes by eigenvector centrality:", sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

# Visualization of Centrality Measures
# Node size by degree centrality and color by betweenness centrality
node_size = [v * 10000 for v in degree_centrality.values()]
node_color = [betweenness_centrality[node] * 100 for node in fb_graph.nodes()]

plt.figure(figsize=(15,15))
pos = nx.spring_layout(fb_graph)
nx.draw(fb_graph, pos=pos, node_size=node_size, node_color=node_color, with_labels=False, cmap=plt.cm.Blues, alpha=0.7)
plt.title("Network Visualization with Centrality Measures")
plt.axis("off")
plt.show()

# --- Community Detection ---
# Using Girvan-Newman community detection algorithm
from networkx.algorithms.community import girvan_newman

communities = girvan_newman(fb_graph)
first_community = next(communities)  # Get the first level of communities
print("Detected communities (sample):", list(first_community)[:5])  # Print sample of first community

# Visualization of communities (using colors to distinguish them)
community_color_map = {}
for i, community in enumerate(first_community):
    for node in community:
        community_color_map[node] = i

community_colors = [community_color_map[node] for node in fb_graph.nodes()]
plt.figure(figsize=(15,15))
nx.draw(fb_graph, pos=pos, node_size=50, node_color=community_colors, with_labels=False, cmap=plt.cm.Set1, alpha=0.7)
plt.title("Network Communities (Girvan-Newman)")
plt.axis("off")
plt.show()

# --- Bridge Detection ---
# Identify bridges (critical edges) in the network
bridges = list(nx.bridges(fb_graph))
print("Number of bridges detected:", len(bridges))

# Visualize the network with bridges highlighted in green
plt.figure(figsize=(15,15))
nx.draw(fb_graph, pos=pos, node_size=50, with_labels=False, alpha=0.7, node_color="skyblue", edge_color="gray")
nx.draw_networkx_edges(fb_graph, pos, edgelist=bridges, edge_color="green", width=2)
plt.title("Network with Bridges Highlighted")
plt.axis("off")
plt.show()

# --- Network Analysis ---
# Clustering coefficient analysis (Average clustering)
avg_clustering = nx.average_clustering(fb_graph)
print(f"Average Clustering Coefficient: {avg_clustering}")

# Network density (ratio of actual edges to possible edges)
network_density = nx.density(fb_graph)
print(f"Network Density: {network_density}")
