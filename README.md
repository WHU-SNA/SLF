# SLF
Python implementation of the method proposed in
"[Link Prediction with Signed Latent Factors in Signed Social Networks](https://dl.acm.org/doi/pdf/10.1145/3292500.3330850)", Pinghua Xu, Wenbin Hu, Jia Wu and Bo Du, SIGKDD 2019.

## Overview
This repository is organised as follows:
- `input/` contains the four graphs `WikiElec` `WikiRfa` `Slashdot` `Epinions` used in the experiments;
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

**NOTE** All the used graphs are **directed**. However, if you want to handle an **undirected** graph, modify your input file to make that each edge (u, v, w) constitutes two rows of the file like the following:

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
**NOTE** As **sign prediction** is a more popular evaluation task, `--link-prediction` is set to `False` and `--sign-prediction` is set to `True` by default. You can refer to our paper to find the difference between the two tasks.

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

If you want to learn node embedding for other use and not to waste time performing link prediction or sign prediction, then run:
```
python src/main.py --link-prediction False --sign-prediction False
```

## Output

### Tasks on signed networks
For **sign prediction** task, we use `AUC` and `Macro-F1` for evaluation.

For **link prediction** task, we use `AUC@p`, `AUC@n` and `AUC@non` for evaluation. Refer to our paper for detailed description. We adimit that it is a wrong choice to use `Micro-F1` for evaluation on a dataset with unbalanced labels, so we removed this metric.

We perform the evaluation after each epoch, and output the provisional result like the following:
```
Epoch 0 Optimizing: 100%|██████████████████████████████████████| 6637/6637 [00:19<00:00, 343.23it/s]
Evaluating...
Sign prediction, epoch 0: AUC 0.832, F1 0.697
Link prediction, epoch 0: AUC@p 0.901, AUC@n 0.750, AUC@non 0.878
Epoch 2 Optimizing: 100%|██████████████████████████████████████| 6637/6637 [00:18<00:00, 349.84it/s]
Evaluating...
Sign prediction, epoch 2: AUC 0.838, F1 0.739
Link prediction, epoch 2: AUC@p 0.885, AUC@n 0.762, AUC@non 0.867
```

When the training is ended up, the evaluation results are printed in tabular format. If `--sign-prediction==True`, the results of sign prediction are printed like the following:
| Epoch | AUC | Macro-F1 |
| :-----:| :----: | :----: |
| 0 | 0.832 | 0.697 |
| 1 | 0.858 | 0.730 |
| 2 | 0.838 | 0.739 |
| ... | ... | ... |
| 19 | 0.905 | 0.802 |

And if `--link-prediction==True`, the results of link prediction are printed like the following:
| Epoch | AUC@p | AUC@n | AUC@non |
| :-----:| :----: | :----: | :----: |
| 0 | 0.901 | 0.750 | 0.878 |
| 1 | 0.882 | 0.739 | 0.855 |
| 2 | 0.885 | 0.762 | 0.867 |
| ... | ... | ... | ... |
| 19 | 0.943 | 0.920 | 0.948 |


### Node embeddings
The learned embeddings are saved in `output/` in `.npz` format (supported by `Numpy`). Note that if the maximal node ID is 36, then the embedding matrix has 36+1 rows ordered by node ID (as the ID can start from 0). Although some nodes may not exist (e.g., node 11 is removed from the original dataset), it does not matter.

You can use them for any purpose in addition to the two performed tasks.

## Baselines
In our paper, we used the following methods for comparison:
- `SIGNet`  "Signet: Scalable embeddings for signed networks" [[source](https://github.com/raihan2108/signet)]
- `MF`      "Low rank modeling of signed networks"
- `LSNE`    "Solving link-oriented tasks in signed network via an embedding approach"
- `SIDE`    "Side: representation learning in signed directed networks" [[source](https://datalab.snu.ac.kr/side/)]

`MF` and `LSNE` are not open-sourced, but if you want our implementation of these methods, email to xupinghua@whu.edu.cn

## Cite
If you find this repository useful in your research, please cite our paper:

```
@inproceedings{xu2019link,
  title={Link prediction with signed latent factors in signed social networks},
  author={Xu, Pinghua and Hu, Wenbin and Wu, Jia and Du, Bo},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1046--1054},
  year={2019}
}
```
