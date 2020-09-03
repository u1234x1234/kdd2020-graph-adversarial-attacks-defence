[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/u1234x1234/kdd2020-graph-adversarial-attacks-defence/blob/master/LICENSE)

# 6th place solution to KDD CUP 2020 Graph Adversarial Attacks & Defence

The final results:

https://www.biendata.xyz/competition/kddcup_2020/winners/

# Usage

To generate the attacker graph & train defender model:
```bash
docker build -t kdd ./
docker run --gpus 0 -it kdd
```

After that 3 files will be available inside the docker image:
* adj.pkl - edges of the attacker graph
* features.npy - nodes of the attacker graph
* model.pkl - fitted defender model


To make defender submission:
```bash
cp model.pkl defender
cd defender && zip -r defender_submission.zip ./*
```

# Description

For the detailed description & intuition please refer to the [attached PDF](kdd-cup2020-adversarial-graph-attack-defence-u1234x1234.pdf)

## Defender

[Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)

```
Stack of the graph convolutional layers
Basic block: ​Simplifying Graph Convolution layer with number of hops=4
Number of layers​: 3
Number of hidden nodes per layer:​ [140, 120, 100]
Optimizer: ​AdamW​ with learning rate ​0.01
Normalization of input features​: BatchNorm
Normalization between Graph Convolutional layers​: LayerNorm
Activation between Graph Convolutional layers​: tanh
Weight decay:​ 0
Loss function: Multi-class classification ​cross entropy with softmax
Number of epochs: 600
```

## Attacker

Attacker nodes with fixed features attached to the nodes (of the original graph) with a small degree.

# Acknowledgements

* DGL library https://github.com/dmlc/dgl - great library
* PyTorch geometric https://github.com/rusty1s/pytorch_geometric - also great library but was not used in the final submission

# Test defender submission

```
docker build -t defe:001 ./

docker run --gpus=0 -it -v /home/u1234x1234/kdd2020-graph-adversarial-attacks-defence/data/kdd_cup_phase_two/:/data defe:001 "/data/adj_matrix_formal_stage.pkl /data/feature_formal_stage.npy /data/output.csv"
```