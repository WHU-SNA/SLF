import argparse
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from texttable import Texttable


def parameter_parser():
    """
    Parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run SLF.")
    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./input/WikiElec.txt",
                        help="Edge list in txt format.")
    parser.add_argument("--outward-embedding-path",
                        nargs="?",
                        default="./output/WikiElec_outward",
                        help="Outward embedding path.")
    parser.add_argument("--inward-embedding-path",
                        nargs="?",
                        default="./output/WikiElec_inward",
                        help="Inward embedding path.")
    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="Number of training epochs. Default is 20.")
    parser.add_argument("--k1",
                        type=int,
                        default=32,
                        help="Dimension of positive SLF. Default is 32.")
    parser.add_argument("--k2",
                        type=int,
                        default=32,
                        help="Dimension of negative SLF. Default is 32.")
    parser.add_argument("--p0",
                        type=float,
                        default=0.001,
                        help="Effect of no feedback. Default is 0.001.")
    parser.add_argument("--n",
                        type=int,
                        default=5,
                        help="Number of noise samples. Default is 5.")
    parser.add_argument("--link-prediction",
                        type=bool,
                        default=False,
                        help="Make link prediction or not. Default is 5.")
    parser.add_argument("--sign-prediction",
                        type=bool,
                        default=True,
                        help="Make sign prediction or not. Default is 5.")
    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
                        help="Test ratio. Default is 0.2.")
    parser.add_argument("--split-seed",
                        type=int,
                        default=16,
                        help="Random seed for splitting dataset. Default is 16.")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.025,
                        help="Learning rate. Default is 0.025.")

    return parser.parse_args()


def fa(x, args):
    """
    Activation function f_a(x).
    """
    if x > 15:
        tmp = 1
    else:
        tmp = args.p0 * np.exp(x) / (1 + args.p0 * (np.exp(x) - 1))
    return tmp


def read_edge_list(args):
    """
    Load edges from a txt file.
    """
    G = nx.DiGraph()
    edges = np.loadtxt(args.edge_path)
    for i in range(edges.shape[0]):
        G.add_edge(int(edges[i][0]), int(edges[i][1]), weight=edges[i][2])
    edges = [[e[0], e[1], e[2]['weight']] for e in G.edges.data()]
    return edges, max(G.nodes) + 1  # index can start from 0.


@ignore_warnings(category=ConvergenceWarning)
def sign_prediction(out_emb, in_emb, train_edges, test_edges):
    """
    Evaluate the performance on the sign prediction task.
    :param out_emb: Outward embeddings.
    :param in_emb: Inward embeddings.
    :param train_edges: Edges for training the model.
    :param test_edges: Edges for test.
    """
    out_dim = out_emb.shape[1]
    in_dim = in_emb.shape[1]
    train_edges = train_edges
    train_x = np.zeros((len(train_edges), (out_dim + in_dim) * 2))
    train_y = np.zeros((len(train_edges), 1))
    for i, edge in enumerate(train_edges):
        u = edge[0]
        v = edge[1]
        if edge[2] > 0:
            train_y[i] = 1
        else:
            train_y[i] = 0
        train_x[i, : out_dim] = out_emb[u]
        train_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        train_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        train_x[i, out_dim * 2 + in_dim:] = in_emb[v]

    test_edges = test_edges
    test_x = np.zeros((len(test_edges), (out_dim + in_dim) * 2))
    test_y = np.zeros((len(test_edges), 1))
    for i, edge in enumerate(test_edges):
        u = edge[0]
        v = edge[1]
        if edge[2] > 0:
            test_y[i] = 1
        else:
            test_y[i] = 0
        test_x[i, : out_dim] = out_emb[u]
        test_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        test_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        test_x[i, out_dim * 2 + in_dim:] = in_emb[v]

    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.fit_transform(test_x)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train_x, train_y.ravel())
    test_y_score = lr.predict_proba(test_x)[:, 1]
    test_y_pred = lr.predict(test_x)
    auc_score = roc_auc_score(test_y, test_y_score, average='macro')
    macro_f1_score = f1_score(test_y, test_y_pred, average='macro')

    return auc_score, macro_f1_score


@ignore_warnings(category=ConvergenceWarning)
def link_prediction(out_emb, in_emb, train_edges, test_edges, train_edges_null, test_edges_null, num_node):
    """
    Evaluate the performance on the link prediction task.
    :param out_emb: Outward embeddings.
    :param in_emb: Inward embeddings.
    :param train_edges: Edges for training the model.
    :param test_edges: Edges for test.
    """
    out_dim = out_emb.shape[1]
    in_dim = in_emb.shape[1]
    train_x = np.zeros((len(train_edges) + len(train_edges_null), (out_dim + in_dim) * 2))
    train_y = np.zeros((len(train_edges) + len(train_edges_null), 1))
    for i, edge in enumerate(train_edges):
        u = edge[0]
        v = edge[1]
        train_x[i, : out_dim] = out_emb[u]
        train_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        train_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        train_x[i, out_dim * 2 + in_dim:] = in_emb[v]
        if edge[2] > 0:
            train_y[i] = 1
        else:
            train_y[i] = -1

    for i, edge in enumerate(train_edges_null):
        i += len(train_edges)
        u = edge[0]
        v = edge[1]
        train_x[i, : out_dim] = out_emb[u]
        train_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        train_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        train_x[i, out_dim * 2 + in_dim:] = in_emb[v]
        train_y[i] = 0

    test_x = np.zeros((len(test_edges) + len(test_edges_null), (out_dim + in_dim) * 2))
    test_y = np.zeros((len(test_edges) + len(test_edges_null), 1))
    for i, edge in enumerate(test_edges):
        u = edge[0]
        v = edge[1]
        test_x[i, : out_dim] = out_emb[u]
        test_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        test_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        test_x[i, out_dim * 2 + in_dim:] = in_emb[v]
        if edge[2] > 0:
            test_y[i] = 1
        else:
            test_y[i] = -1

    for i, edge in enumerate(test_edges_null):
        i += len(test_edges)
        u = edge[0]
        v = edge[1]
        test_x[i, : out_dim] = out_emb[u]
        test_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        test_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        test_x[i, out_dim * 2 + in_dim:] = in_emb[v]
        test_y[i] = 0

    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.fit_transform(test_x)
    lr = LogisticRegressionCV(fit_intercept=True, max_iter=100, multi_class='multinomial', Cs=np.logspace(-2, 2, 20),
                              cv=2, penalty="l2", solver="lbfgs", tol=0.01)
    lr.fit(train_x, train_y.ravel())
    pred_prob = lr.predict_proba(test_x)
    labels = test_y.copy()
    for i in range(len(labels)):
        if labels[i] == 1:
            labels[i] = 1
        else:
            labels[i] = 0
    auc_score_pos = roc_auc_score(labels, pred_prob[:, 2])
    labels = test_y.copy()
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = 1
        else:
            labels[i] = 0
    auc_score_neg = roc_auc_score(labels, pred_prob[:, 0])
    labels = test_y.copy()
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = 1
        else:
            labels[i] = 0
    auc_score_null = roc_auc_score(labels, pred_prob[:, 1])

    return auc_score_pos, auc_score_neg, auc_score_null


def args_printer(args):
    """
    Print the parameters in tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable()
    l = [[k, args[k]] for k in args.keys()]
    l.insert(0, ["Parameter", "Value"])
    t.add_rows(l)
    print(t.draw())


def sign_prediction_printer(logs):
    """
    Print the performance on sign prediction task in tabular format.
    :param logs: Logs about the evaluation.
    """
    t = Texttable()
    epoch_list = logs['epoch']
    auc_list = logs['sign_prediction_auc']
    macrof1_list = logs['sign_prediction_macro_f1']
    l = [[epoch_list[i], auc_list[i], macrof1_list[i]] for i in range(len(epoch_list))]
    l.insert(0, ['Epoch', 'AUC', 'Macro-F1'])
    t.add_rows(l)
    print(t.draw())


def link_prediction_printer(logs):
    """
    Print the performance on link prediction task in tabular format.
    :param logs: Logs about the evaluation.
    """
    t = Texttable()
    epoch_list = logs['epoch']
    auc_p_list = logs['link_prediction_auc@p']
    auc_n_list = logs['link_prediction_auc@n']
    auc_non_list = logs['link_prediction_auc@non']
    l = [[epoch_list[i], auc_p_list[i], auc_n_list[i], auc_non_list[i]] for i in range(len(epoch_list))]
    l.insert(0, ['Epoch', 'AUC@p', 'AUC@n', 'AUC@non'])
    t.add_rows(l)
    print(t.draw())
