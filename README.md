# SLF
Python implementation of the method proposed in the paper:
"[Link Prediction with Signed Latent Factors in Signed Social Networks](https://dl.acm.org/doi/pdf/10.1145/3292500.3330850)", Pinghua Xu, Wenbin Hu, Jia Wu and Bo Du, SIGKDD 2019.

## Overview
This repository is organised as follows:
- `input/` contains the four datasets downloaded used in the experiments;
- `output/` is the directory to store the learned node embeddings;
- `src/` contains the implementation of the proposed SLF method.

## Requirements
The implementation is tested under Python 3.7, with the folowing packages installed:
- `networkx==2.3`
- `numpy==1.16.5`
- `scikit-learn==0.21.3`
- `texttable==1.6.2`
- `tqdm==4.36.1`

## Input
The code takes an input graph in `txt` format. Every row indicates an edge between two nodes separated by a `space` or `\t`. The file does not contain a header. Nodes can be indexed starting with any non-negative number. Four sample graphs (donwloaded from [SNAP](http://snap.stanford.edu/data/#signnets), but node ID is resorted) `WikiElec`, `WikiRfa`, `Slashdot` and `Epinions` are included in the `input/` directory. The structure of the input file is the following:

| Source node | Target node | Weight |
| :-----:| :----: | :----: |
| 0 | 1 | -1 |
| 1 | 3 | 1 |
| 1 | 2 | 1 |
| 2 | 4 | -1 |

**NOTE** All the used graphs are *directed*. However, if you want to handle an *undirected* graph, modify your input file to make that each edge (u, v, w) constitutes two rows of the file like the following:

| Source node | Target node | Weight |
| :-----:| :----: | :----: |
| u | v | w |
| v | u | w |

## Options
### Input and output options
```
--edge-path                 STR    Input file path           Default=="./input/WikiElec.txt"
--outward-embedding-path    STR    Outward embedding path    Default=="./output/WikiElec_outward"
--inward-embedding-path     STR    Inward embedding path     Default=="./output/WikiElec_inward"
```
### Model options
```
--epochs          INT     Number of training epochs    Default==20
--k1              INT     Positive SLF dimension       Default==32
--k2              INT     Negative SLF dimension       Default==32
--p0              FLOAT   Effect of no feedback        Default==0.001
--n               INT     Number of noise samples      Default==5
--learning-rate   FLOAT   Leaning rate                 Default==0.025
```
### Evaluation options
```
--test-size          FLOAT    Test ratio                           Default==0.2
--split-seed         INT      Random seed for splitting dataset    Default==16
--link-prediction    BOOL     Make link prediction or not          Default=False
--sign-prediction    BOOL     Make sign prediction or not          Default=True
```
**NOTE** As sign-prediction is more popular evaluation task, `--link-prediction` is set to `False` and `--sign-prediction` is set to `True` by default. You can refer to our paper to find the difference between the two tasks.

## Examples
Train an SLF model on the deafult `WikiElec` dataset, output the performance on sign prediction task, and save the embeddings:
```
python src/main.py
```
Train an SLF model with custom epoch number and test ratio:
```
python src/main.py --epochs 30 --test-size 0.3
```

Train an SLF model on the `WikiRfa` dataset, perform link prediction task but not sign prediction task:
```
python src/main.py --edge-path ./input/WikiRfa.txt --outward-embedding-path ./output/WikiElec_outward --inward-embedding-path ./output/WikiElec_inward --link-prediction True --sign-prediction False
```

## Cite
If you find the code useful in your research, please cite the original paper:

```
@inproceedings{xu2019link,
  title={Link prediction with signed latent factors in signed social networks},
  author={Xu, Pinghua and Hu, Wenbin and Wu, Jia and Du, Bo},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1046--1054},
  year={2019}
}
```

## Note
Except for link prediction, the node representations learned by SLF achieve state-of-the-art performance in sign prediction task as well.
