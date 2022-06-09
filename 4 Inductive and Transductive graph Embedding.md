## Review GNN Embedding methods

How to learn parameters?
Supervised: Softmax with the true groud to get loss, minimize the loss
Unsupervised: sam class dist close, diff class dist far
Semi supervised: 

Why GNN Embedding methods better?
- have parameters sharing, unlike deep walk and Node2Vec
- used features, unlike deep walk and Node2Vec

How better?
- Transform unseen node or unseen edge
- Transduction -> Inductive

**Inductive Models**
Get embedding vectors for unseen nodes

Learn a func that generates embedding by aggregating features from a node's local neiborhood

**Search depth**? consider neibors with how far?
**Smapling** neibor instead of using all

MPF (Message Passing Framework)

## GraphSAGE

Aggregators:
- Mean
- LSTM
- Pooling

