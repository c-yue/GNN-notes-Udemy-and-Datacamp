# CNN to GNN

## Convolution Operation

1. why
Embedding expensive, miss node features, connot predict unseen nodes  
-> CNN
2. how apply convolution to a graph representation?
3. review CNN
CNN in 2D image: kernel to learn -   
eg.  
every original point -> get its neigbor ie. 3*3 unit  
-> point multi with kernel -> the result number
-> the result number represent the feature of the original points
-> repeat to get feature of all points (strides)
-> kernel can be learned
PS:  
point multi:  
product elements in the same position, sum them, result a number
4. benefits
kernel parameters sharing by pixels  
translation invariant: 旋转图片不会带来过大影响  
number of parameters independent of input images  
5. challenge to use in graph
number of heighbors changes  
distance between nodes changes, which should be considered in graph context  
number of attributes can vary  
diff nodes have diff meaning, eg. bipartite
&nbsp;



## Graph Convolution

1. graph in signal processing & covid-19  
node: country  
feature: population, covid_number  
edge: distance  
2. challenge with Neural Network, not scalable -> CNN works
3. challenge with CNN, graph structure irregular -> GCN works
4. GCN in signal processing  
    - 4 signals with the same time diff  
    - 4 with diff distance  
    - 4 nodes with features  
    - S as shift matrix, Marcov Process   
    - **S <=> adjacency matrix** - 转移矩阵即为邻接矩阵  
    - S can be more complex, not only 0, 1  
    - wight*S is the kernal, compared to small kernal matrix in CNN  
5. a layer:
    - 原本的图，经过swift，再经过激活，即得到新图  
    - G' = sum of (weightk * Sk) = sum of weighted shift   
    - x' = G' 激活  
    - $G' = w_0*S^0*x + w_1*S^1*x + w_2*S^2*x + ... + w_k*S^k*x$  
    - $x' = \sigma(G'*x + b)$  

6. why need to shift for k times?  
-> $w_0*S^0*x$: the existing nodes  
-> more and more info to the nodes, for specific node, it have global info  
-> only $w_0, w_1, w_2, ... w_k$ makes the features of a node diff  
-> as for the S, it is static from the beginning  

7. graph frequency response

&nbsp;



## Message Passing Framework
1. math frame of GNN operation
GNN operation - 比如某层GCN
2. Neural Massage Passing
- initialization
    - graph, nodes
    - $h_v$: hidden states, $h_v = x_v$
- **aggregation** the node with neighbor nodes
    - how other nodes influent the existing node
    - $a_v = f_{aggregate}()$
    - methods: 很多，下一章介绍
- **update** current node with neighbor nodes
    - $h_{v+1} = f_{update}(m_v, h_v)$



