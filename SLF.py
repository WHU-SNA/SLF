import networkx as nx
import numpy as np
import random as rd
from tqdm import tqdm


def fa(x):
    if x > 15:
        tmp = 1
    else:
        tmp = p0 * np.exp(x) / (1 + p0 * (np.exp(x) - 1))
    return tmp


dir_path = 'Elec/80/'
dim = 32
learning_rate_0 = 0.025
noise_size = 5
max_iter = 10
p0 = 0.001

G = nx.DiGraph()
with open(dir_path + 'edges_train.txt') as f:
    for line in f:
        line = line.strip().split('\t')
        if int(line[2]) != 0:
            G.add_edge(int(line[0]), int(line[1]), weight=int(line[2]))
max_node_id = max(G.nodes())
nodes = list(G.nodes())

W_out_p = np.matrix(np.zeros((max_node_id + 1, dim)), dtype=np.float32)
W_in_p = np.matrix(np.zeros((max_node_id + 1, dim)), dtype=np.float32)
W_out_n = np.matrix(np.zeros((max_node_id + 1, dim)), dtype=np.float32)
W_in_n = np.matrix(np.zeros((max_node_id + 1, dim)), dtype=np.float32)
for i in nodes:
    for j in range(dim):
        W_out_p[i, j] = rd.uniform(0, 1)
        W_in_p[i, j] = rd.uniform(0, 1)
        W_out_n[i, j] = rd.uniform(0, 1)
        W_in_n[i, j] = rd.uniform(0, 1)


for count in range(max_iter):
    pbar = tqdm(total=G.number_of_nodes(), desc='Iter_'+str(count)+' Optimizing')
    learning_rate = learning_rate_0 * (max_iter - count) / max_iter
    for u in nodes:
        pbar.update(1)
        out_p_g = np.zeros((1, dim))
        out_n_g = np.zeros((1, dim))
        in_p_g = np.zeros((1, dim))
        in_n_g = np.zeros((1, dim))

        succs = G.successors(u)
        for succ in succs:
            e_p = fa(W_out_p[u] @ W_in_p[succ].T)
            e_n = fa(W_out_n[u] @ W_in_n[succ].T)
            if G[u][succ]['weight'] == 1:
                out_p_g += (1 - e_p) * W_in_p[succ]
                out_n_g -= e_n * W_in_n[succ]
            elif G[u][succ]['weight'] == -1:
                out_p_g -= e_p * W_in_p[succ]
                out_n_g += (1 - e_n) * W_in_n[succ]
            elif G[u][succ]['weight'] == 0:
                out_p_g += (1 - e_p) * W_in_p[succ]
                out_n_g += (1 - e_n) * W_in_n[succ]
        for i in range(noise_size):
            noise = rd.choice(nodes)
            while noise in succs:
                noise = rd.choice(nodes)
            e_p = fa(W_out_p[u] @ W_in_p[noise].T)
            e_n = fa(W_out_n[u] @ W_in_n[noise].T)
            out_p_g -= e_p * W_in_p[noise]
            out_n_g -= e_n * W_in_n[noise]

        pres = G.predecessors(u)
        for pre in pres:
            e_p = fa(W_out_p[pre] @ W_in_p[u].T)
            e_n = fa(W_out_n[pre] @ W_in_n[u].T)
            if G[pre][u]['weight'] == 1:
                in_p_g += (1 - e_p) * W_out_p[pre]
                in_n_g -= e_n * W_out_n[pre]
            elif G[pre][u]['weight'] == -1:
                in_p_g -= e_p * W_out_p[pre]
                in_n_g += (1 - e_n) * W_out_n[pre]
            elif G[pre][u]['weight'] == 0:
                in_p_g += (1 - e_p) * W_out_p[pre]
                in_n_g += (1 - e_n) * W_out_n[pre]
        for i in range(noise_size):
            noise = rd.choice(nodes)
            while noise in pres:
                noise = rd.choice(nodes)
            e_p = fa(W_out_p[noise] @ W_in_p[u].T)
            e_n = fa(W_out_n[noise] @ W_in_n[u].T)
            in_p_g -= e_p * W_out_p[noise]
            in_n_g -= e_n * W_out_n[noise]

        W_out_p[u] += learning_rate * out_p_g
        W_in_p[u] += learning_rate * in_p_g
        W_out_n[u] += learning_rate * out_n_g
        W_in_n[u] += learning_rate * in_n_g

        for i in range(dim):
            if W_out_p[u, i] < 0:
                W_out_p[u, i] = 0
            if W_in_p[u, i] < 0:
                W_in_p[u, i] = 0
            if W_out_n[u, i] < 0:
                W_out_n[u, i] = 0
            if W_in_n[u, i] < 0:
                W_in_n[u, i] = 0
    pbar.close()

W_out = np.matrix(np.zeros((max_node_id + 1, dim * 2)), dtype=np.float32)
W_in = np.matrix(np.zeros((max_node_id + 1, dim * 2)), dtype=np.float32)
for i in range(max_node_id + 1):
    W_out[i, : dim] = W_out_p[i]
    W_out[i, dim: dim * 2] = W_out_n[i]
    W_in[i, : dim] = W_in_p[i]
    W_in[i, dim: dim * 2] = W_in_n[i]

np.save(dir_path + 'outEmb', W_out)
np.save(dir_path + 'inEmb', W_in)
