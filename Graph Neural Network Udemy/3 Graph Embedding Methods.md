# Graph Embedding Methods

## Graph Embedding Problem Statement

Embedding
- Encoder  
- **similarity** of the nodes and the embedding nodes **almost the same**  
- $S_G$ => $S_E$ - old space to new space  
- **Message Passing Framework** is a Embedding method, Encoder, Both SGC and GCN are based on this method
- other methods: deepwalk, popular before GNN

Diff Levels of Embedding
- Node embedding
- Edge embedding
- Global embedding

Data Format (Input for embedding)
- nodes as list
- edges as list
- Adjacency list
- 这种存储方式数据密集，且可以调换节点的存储顺序，只需要在edges及adjacency进行调整即可


## Deep Walk
**Goal**: $S_G(u,v)$ => $S_E(z_u, z_v)$  

**Similarity**:  
- $*S_G(u, v)* = p(u|v)$  
    - 从v开始，random walk  
    => low chance to node u  
    => S_G(u,v) = p(u|v) low  
    => similarity low  

- $S_E(z_u, z_v) = exp(z_u^Tz_v)/ \sum_Vexp(z_u^Tz_u)$  
    - dot product of the two vectors devided by sum of vectors products

**Loss**: $l = \underset {(u,v)\in D} {\sum} -log(S_E(z_u, z_v))$

**T**: steps to take in a random walk

**d**: 每个点都有一个映射，d dimensions, **d** 可自行定义

**Hyper Parameters**: **T** steps, **d** dimensions in embedding space

```
" Workshop on DeepWalk Algorithm using Karate Club"
import networkx as nx
from karateclub import DeepWalk 
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


G = nx.karate_club_graph() # load the Zachary's karate club graph
print('Number of nodes (club-members)', len(G.nodes))
nx.draw_networkx(G)

" plot the graph with labels"
labels = []
for i in G.nodes:
    club_names = G.nodes[i]['club']
    labels.append(1 if club_names == 'Officer' else 0) #Clubs: 'Officer' or 'Mr.Hi'
    
layout_pos = nx.spring_layout(G)
nx.draw_networkx(G,pos = layout_pos ,node_color = labels, cmap='coolwarm')

" Perform node embedding using DeepWalk "
Deepwalk_model = DeepWalk(walk_number=10, walk_length=80, dimensions=124)
Deepwalk_model.fit(G)
embedding = Deepwalk_model.get_embedding()
print('Embedding array shape (nodes x features):',embedding.shape )

" Low dimensional plot of the neodes x features"

PCA_model = sklearn.decomposition.PCA(n_components=2)
lowdimension_embedding = PCA_model.fit_transform(embedding)
print('Low dimensional embedding representaiton (nodes x 2):', lowdimension_embedding.shape)
plt.scatter(lowdimension_embedding[:,0],lowdimension_embedding[:,1],c=labels,
            s=15,cmap='coolwarm')


" Node classification using embedded model"
x_train,x_test,y_train,y_test = train_test_split(embedding, labels, test_size=0.3)
ML_model = LogisticRegression(random_state=0).fit(x_train,y_train)
y_predict = ML_model.predict(x_test)
ML_acc = roc_auc_score(y_test,y_predict)
print('AUC:',ML_acc)
```



## Node2Vec

- random walk -> walk with strategy
- Depend by walk length
- Depth first search(DFS): more global/exploration - q1 小
- Breadth first search(BFS): more local - p2 小

process
-  define graph 
- define p,q
- do sampling to get series
- skipGram to sampled series
- get embedding model and embedding of nodes

embedding of edges
- average of nodes / sum, max

```
" Some codes are the same as the previous code "
" Perform node embedding using Node2Vec "
N2Vec_model = DeepWalk(walk_number=10, walk_length=80, p=0.6, q=0.4, dimensions=124)
N2Vec_model.fit(G)
N2Vec_embedding = N2Vec_model.get_embedding()
print('Embedding array shape (nodes x features):',embedding.shape )

PCA_model = sklearn.decomposition.PCA(n_components=2)
lowdimension_n2vembedding = PCA_model.fit_transform(N2Vec_embedding)
print('Low dimensional embedding representaiton (nodes x 2):', lowdimension_n2vembedding.shape)
plt.scatter(lowdimension_n2vembedding[:,0],lowdimension_n2vembedding[:,1],c=labels,
            s=15,cmap='coolwarm')
```

```
pytorch and geometric method:
Workshop+-+Node2Vec_TorchGeo.py
```






## GNN

Deppwalk和Node2Vec 的不足
- no parameter sharing: 每个node有自己的walk和计算，computation expensive
- no semantic info: Feature nodes not considered
- Not inductive: cannnot predict embedding for unseen data

**Definition**
a transformation on attributes of the graph (including nodes, edges, global graph), not change the connection relationship between nodes

Input a vector, ouput a vector (both are node features / edge features)

**eg (simple)**. MLP to transform node vector to a new node vector
diff nodes use the same MLP
too simple -> use Message Passing Framework

**Message Passing Framework**
Both SGC and GCN are based on this method
Nodes get info from neibors and are fed into a MLP (1 layer)

What to do after the transformation: ML for classfication based on node vectors

**Pooling**
目的1：使用edge算node，使用node算edge，可以补充空信息的顶点
目的2：互相更新，更高效使用图的信息
先后顺序？可以直接交互更新



## SGC： Simplifying Graph Convolution Network

SGC drawbacks:
- all neighbors are treated equally, not weighted
- The shifting $\widetilde{S}^k$ is based on averaging, while averaging maybe not right

```
Workshop+-+SGC.py
```





## GCN:  Graph Convolution Network

Diff to SGC: 
SGC use K step iteration, using the K-1 state
GCN use W and n-1 Layers 

W project matrix, project features into a new space, linear projection
100 dimensions to 50, 



How to update the graph
$H^{l+1} = {\sigma} (\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^l W^l) $

- where 
    - H表示特征矩阵，各个节点的特征
    - $\hat{A} = A  + I$, 表示对节点及节点周边进行相加
    - A: Adjcency Matrix
    - I: Identity Matrix, 单位矩阵
    - $\hat{D} = \sum_j \hat{A}_{ij}$, 即对$\hat{A}$按行求和，将和放在对角线上，度矩阵，表示节点的度
    - $\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$, 在每一层固定不变，实际是节点加和后的归一化
    - ${\sigma}$, Relu sigmoid etc