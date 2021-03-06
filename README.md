# GraphSAGE: Inductive Representation Learning on Large Graphs

Paper Reference: https://arxiv.org/abs/1706.02216

### Install the requirement libraries

```git
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
!pip install -q torch-geometric
```

### Clone the repository

```git
!git clone https://github.com/Ahmed-A-A-Elhag/GraphSAGE
```

### Train the model of Cora dataset

```git
!python GraphSAGE/src/main.py  --num-epochs 200 --data 'Cora' --num-hid  1024 --aggregator 'Mean'
```

### Results


|    Dataset      |          Type  | Nodes          |Edges         | Classes     | Features   |Mean Aggregator    |      MaxPooling Aggregator   |
| -------------   | -------------    |------------- |------------- |------------- |------------- |-------------------- | -------------------- |
| Cora            |  Citation Network |     2,708   |   5,429      |     7        |    1,433     |       80.0 士 1     |    81.0 士 1 |
| Citeseer        |  Citation Network |     3,327   |   4,732      |     6        |    3,703     |       68 士  1   |    69.5 士  0.8| 
