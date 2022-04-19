# Graph Reprensentation  
&nbsp;

## Graph definition

1. nodes, edges, features  
2. directed, undirected  
3. G = (V, E, u)  
V, Nodes  
E, Edges(Adjacency, Weight)  
u, feature vector  
&nbsp;

## Storing graph information
eg. Homogeneous graph  
1. node, edge list 
2. adjacency matrix
Aij from node i to node j
for homogeneous, always square
if undirected, always sysmetric
3. weight matrix
use weight instead of 1,0  
&nbsp;

## Graph degree and laplacian of graph
1. Degree matrix
diagonal matrix  
high degree of a node is more important
2. Laplacian matrix
Lg = Dg - Ag/Wg  
Lg: laplacian matrix  
Dg: degree matrix  
Ag: adjacency matrix  
Wg: weight matrix  
Lg 半正定矩阵, includes  
3. normalized graphs
A bar  
L bar  
&nbsp;

## Learning in graph representation learning
1. node prediction
2. link prediction
3. graph prediction
4. make it simpler
endoding = latent representation  
embedding the graph to anoghe space  
make two nodes similarity almost the same after embedding
decoding: func describe similarity of two points in the embedding space
5. embedding method
5.1 meaning of similarity: Sg in original space, SE in embedding space  
goal: 
5.2 Similarity(u,v) =~ SimilarityE(Zu,Zv)
Sg(u,v) =~ SE(Zu,Zv)  
5.3 loss:
Eucliean distance of every two points  
dis = SE(Zu, Zv) - Sg(u, v)
sum dis sqaure
6. notes used later
all use encoding or decoding methods to find the loss
6.1 matrix factorization = inner product of Zu.T and Zv
6.2 look-up table = inner product of Zu.T and Zv
6.3 random walk: decode statistics of random walks
&nbsp;

## drawbacks
no para sharing: expensive  
no semantic information: node 的特征不被考虑  
not inductive: not predict embedding for unseen data -> so CNN might better  
&nbsp;


## workshop - using torch and torch geometric for define graph
```
# https://www.udemy.com/course/graph-neural-network/learn/lecture/26621590#overview

import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx

" Define a graph "
# a graph with 4 nodes
edge_list = torch.tensor([
                         [0, 0, 0, 1, 2, 2, 3, 3], # Source Nodes
                         [1, 2, 3, 0, 0, 3, 2, 0]  # Target Nodes
                        ], dtype=torch.long)
                        # long -> long float

# 6 Features for each node (4x6 - Number of nodes x NUmber of features)
node_features = torch.tensor([
                            [-8, 1, 5, 8, 2, -3], # Features of Node 0
                            [-1, 0, 2, -3, 0, 1], # Features of Node 1
                            [1, -1, 0, -1, 2, 1], # Features of Node 2
                            [0, 1, 4, -2, 3, 4], # Features of Node 3
                            ],dtype=torch.long)

# 1 Weight for each edge 
edge_weight = torch.tensor([
                            [35.], # Weight for nodes (0,1)
                            [48.], # Weight for nodes (0,2)
                            [12.], # Weight for nodes (0,3)
                            [10.], # Weight for nodes (1,0)
                            [70.], # Weight for nodes (2,0)
                            [5.], # Weight for nodes (2,3)
                            [15.], # Weight for nodes (3,2)
                            [8.], # Weight for nodes (3,0)   
                            ],dtype=torch.long)  

# Make a data object to store graph informaiton 
data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_weight)

" Print the graph info "
print("Number of nodes: ", data.num_nodes)
print("Number of edges: ",data.num_edges)
print("Number of features per node (Length of feature vector): ", data.num_node_features,"\n")
print("Number of weights per edge (edge-features): ", data.num_edge_features, "\n")

" Plot the graph "
G = to_networkx(data)
nx.draw_networkx(G)
```
&nbsp;





























