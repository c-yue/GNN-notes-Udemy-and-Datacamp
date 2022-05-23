# Graph Embedding Methods

## Graph Embedding Problem Statement

Embedding
- encoder  
- **similarity** of the nodes and the embedding nodes **almost the same**  
- $S_G$ => $S_E$ - old space to new space  
- Message Passing Framework is a Embedding, Encoder
- other methods:
- deepwalk, popular before GNN

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

random walk
- Depend by walk length
- Depth first search(DFS): more global - q1
- Breadth first search(BFS): more local - q2F

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



## GNN Motivation

Deppwalk和Node2Vec 的不足
- no parameter sharing: 每个node有自己的walk和计算，computation expensive
- no semantic info: Feature nodes not considered
- Not inductive: cannnot predict embedding for unseen data




## SGC： Simplifying Graph Convolution Network

SGC drawbacks:
- all neighbors are treated equally, not weighted
- The shifting $\widetilde{S}^k$ is based on averaging, while averaging maybe not right


```
Workshop+-+SGC.py
```