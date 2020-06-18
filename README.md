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
The code takes an input graph in `txt` format. Every row indicates an edge between two nodes separated by a `space` or `\t`. The file does not contain a header. Nodes can be indexed starting with any non-negative number. Four sample datasets (donwloaded from [SNAP](http://snap.stanford.edu/data/#signnets), but node ID is resorted) `WikiElec`, `WikiRfa`, `Slashdot` and `Epinions` are included in the `input/` directory. The structure of the input file is the following:

| Source node | Target node | Weight |
| :-----:| :----: | :----: |
| 0 | 1 | -1 |
| 1 | 3 | 1 |
| 1 | 2 | 1 |
| 2 | 4 | -1 |

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
