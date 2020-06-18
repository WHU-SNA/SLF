import networkx as nx
import numpy as np
import random as rd
from tqdm import tqdm
from utils import fa, read_edge_list, sign_prediction, link_prediction
from sklearn.model_selection import train_test_split
import networkx as nx


class SignedLatentFactorModel(object):
    def __init__(self, args):
        self.args = args
        self.logs = {'epoch': [], 'sign_prediction_auc': [], 'sign_prediction_macro_f1': [],
                     'link_prediction_auc@p': [], 'link_prediction_auc@n': [], 'link_prediction_auc@non': []}
        self.setup()

        # Positive outward SLF vectors.
        self.W_out_p = np.matrix(np.zeros((self.num_node, self.args.k1)), dtype=np.float32)
        # Positive inward SLF vectors.
        self.W_in_p = np.matrix(np.zeros((self.num_node, self.args.k1)), dtype=np.float32)
        # Negative outward SLF vectors.
        self.W_out_n = np.matrix(np.zeros((self.num_node, self.args.k2)), dtype=np.float32)
        # Negative inward SLF vectors.
        self.W_in_n = np.matrix(np.zeros((self.num_node, self.args.k2)), dtype=np.float32)

    def setup(self):
        self.edges, self.num_node = read_edge_list(self.args)
        self.train_edges, self.test_edges, = train_test_split(self.edges,
                                                              test_size=self.args.test_size,
                                                              random_state=self.args.split_seed)

        # Generate null links set for link prediction task.
        if self.args.link_prediction:
            G = nx.DiGraph()
            G.add_nodes_from(range(self.num_node))
            G.add_edges_from([[e[0], e[1]] for e in self.edges])
            self.train_edges_null, self.test_edges_null = [], []
            for _ in range(3 * len(self.test_edges)):
                u = rd.choice(range(self.num_node))
                v = rd.choice(range(self.num_node))
                while v in list(G.successors(u)):
                    v = rd.choice(range(self.num_node))
                self.test_edges_null.append([u, v, 'n'])
            for _ in range(3 * len(self.train_edges)):
                u = rd.choice(range(self.num_node))
                v = rd.choice(range(self.num_node))
                while v in list(G.successors(u)):
                    v = rd.choice(range(self.num_node))
                self.train_edges_null.append([u, v, 'n'])


    def fit(self):
        """
        Learn node embeddings.
        """
        G = nx.DiGraph()
        for edge in self.train_edges:
            G.add_edge(edge[0], edge[1], weight=edge[2])
        nodes = list(G.nodes())

        for i in nodes:
            for j in range(self.args.k1):
                self.W_out_p[i, j] = rd.uniform(0, 1)
                self.W_in_p[i, j] = rd.uniform(0, 1)

        for i in nodes:
            for j in range(self.args.k2):
                self.W_out_n[i, j] = rd.uniform(0, 1)
                self.W_in_n[i, j] = rd.uniform(0, 1)

        for epoch in range(self.args.epochs):
            pbar = tqdm(total=G.number_of_nodes(), desc='Epoch ' + str(epoch) + ' Optimizing', ncols=100)
            learning_rate = self.args.learning_rate * (self.args.epochs - epoch) / self.args.epochs
            for u in nodes:
                pbar.update(1)
                out_p_g = np.zeros((1, self.args.k1))
                out_n_g = np.zeros((1, self.args.k2))
                in_p_g = np.zeros((1, self.args.k1))
                in_n_g = np.zeros((1, self.args.k2))

                succs = G.successors(u)
                for succ in succs:
                    e_p = fa(self.W_out_p[u] @ self.W_in_p[succ].T, self.args)
                    e_n = fa(self.W_out_n[u] @ self.W_in_n[succ].T, self.args)
                    if G[u][succ]['weight'] == 1:
                        out_p_g += (1 - e_p) * self.W_in_p[succ]
                        out_n_g -= e_n * self.W_in_n[succ]
                    elif G[u][succ]['weight'] == -1:
                        out_p_g -= e_p * self.W_in_p[succ]
                        out_n_g += (1 - e_n) * self.W_in_n[succ]
                    elif G[u][succ]['weight'] == 0:
                        out_p_g += (1 - e_p) * self.W_in_p[succ]
                        out_n_g += (1 - e_n) * self.W_in_n[succ]
                for i in range(self.args.n):
                    noise = rd.choice(nodes)
                    while noise in succs:
                        noise = rd.choice(nodes)
                    e_p = fa(self.W_out_p[u] @ self.W_in_p[noise].T, self.args)
                    e_n = fa(self.W_out_n[u] @ self.W_in_n[noise].T, self.args)
                    out_p_g -= e_p * self.W_in_p[noise]
                    out_n_g -= e_n * self.W_in_n[noise]

                pres = G.predecessors(u)
                for pre in pres:
                    e_p = fa(self.W_out_p[pre] @ self.W_in_p[u].T, self.args)
                    e_n = fa(self.W_out_n[pre] @ self.W_in_n[u].T, self.args)
                    if G[pre][u]['weight'] == 1:
                        in_p_g += (1 - e_p) * self.W_out_p[pre]
                        in_n_g -= e_n * self.W_out_n[pre]
                    elif G[pre][u]['weight'] == -1:
                        in_p_g -= e_p * self.W_out_p[pre]
                        in_n_g += (1 - e_n) * self.W_out_n[pre]
                    elif G[pre][u]['weight'] == 0:
                        in_p_g += (1 - e_p) * self.W_out_p[pre]
                        in_n_g += (1 - e_n) * self.W_out_n[pre]
                for i in range(self.args.n):
                    noise = rd.choice(nodes)
                    while noise in pres:
                        noise = rd.choice(nodes)
                    e_p = fa(self.W_out_p[noise] @ self.W_in_p[u].T, self.args)
                    e_n = fa(self.W_out_n[noise] @ self.W_in_n[u].T, self.args)
                    in_p_g -= e_p * self.W_out_p[noise]
                    in_n_g -= e_n * self.W_out_n[noise]

                self.W_out_p[u] += learning_rate * out_p_g
                self.W_in_p[u] += learning_rate * in_p_g
                self.W_out_n[u] += learning_rate * out_n_g
                self.W_in_n[u] += learning_rate * in_n_g

                for i in range(self.args.k1):
                    if self.W_out_p[u, i] < 0:
                        self.W_out_p[u, i] = 0
                    if self.W_in_p[u, i] < 0:
                        self.W_in_p[u, i] = 0
                for i in range(self.args.k2):
                    if self.W_out_n[u, i] < 0:
                        self.W_out_n[u, i] = 0
                    if self.W_in_n[u, i] < 0:
                        self.W_in_n[u, i] = 0
            pbar.close()

            W_out = np.matrix(np.zeros((self.num_node, self.args.k1 + self.args.k2)), dtype=np.float32)
            W_in = np.matrix(np.zeros((self.num_node, self.args.k1 + self.args.k2)), dtype=np.float32)
            for i in range(self.num_node):
                W_out[i, : self.args.k1] = self.W_out_p[i]
                W_out[i, self.args.k1:] = self.W_out_n[i]
                W_in[i, : self.args.k1] = self.W_in_p[i]
                W_in[i, self.args.k1:] = self.W_in_n[i]
            print('Evaluating...')
            if self.args.sign_prediction:
                auc, f1 = sign_prediction(W_out, W_in, self.train_edges, self.test_edges)
                self.logs['epoch'].append(epoch)
                self.logs['sign_prediction_auc'].append(auc)
                self.logs['sign_prediction_macro_f1'].append(f1)
                print('Sign prediction, epoch %d: AUC %.3f, F1 %.3f' % (epoch, auc, f1))
            if self.args.link_prediction:
                auc_p, auc_n, auc_null = link_prediction(W_out, W_in, self.train_edges, self.test_edges,
                                                         self.train_edges_null, self.test_edges_null, self.num_node)
                self.logs['link_prediction_auc@p'].append(auc_p)
                self.logs['link_prediction_auc@n'].append(auc_n)
                self.logs['link_prediction_auc@non'].append(auc_null)
                print(
                    'Link prediction, epoch %d: AUC@p %.3f, AUC@n %.3f, AUC@non %.3f' % (epoch, auc_p, auc_n, auc_null))


    def save_emb(self):
        """
        Save the node embeddings in npz format.
        """
        W_out = np.matrix(np.zeros((self.num_node, self.args.k1 + self.args.k2)), dtype=np.float32)
        W_in = np.matrix(np.zeros((self.num_node, self.args.k1 + self.args.k2)), dtype=np.float32)
        for i in range(self.num_node):
            W_out[i, : self.args.k1] = self.W_out_p[i]
            W_out[i, self.args.k1:] = self.W_out_n[i]
            W_in[i, : self.args.k1] = self.W_in_p[i]
            W_in[i, self.args.k1:] = self.W_in_n[i]

        np.save(self.args.outward_embedding_path, W_out)
        np.save(self.args.inward_embedding_path, W_in)
