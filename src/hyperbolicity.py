import os
import random
import pickle as pkl
import sys
import time

import networkx as nx
import numpy as np
from tqdm import tqdm

from torch_geometric.datasets import Planetoid, ZINC
from torch_geometric.utils import to_dense_adj, to_networkx
from data_utils import load_data_lp, load_data_nc

def hyperbolicity_sample(G, num_samples=5000):
    curr_time = time.time()
    hyps = []
    # for i in tqdm(range(num_samples)):
    for i in range(num_samples):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    
    # print('Time for hyp: ', time.time() - curr_time)
    return max(hyps)

# pubmed = Planetoid("./bin/", "PubMed")
# print (pubmed)

zinc = ZINC("./bin/")
print (zinc)

def convert_zinc_to_nx(data):
    g = to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"])
    return nx.from_scipy_sparse_matrix(nx.adjacency_matrix(g, nodelist=list(range(data.x.size(0)))))

N_graphs = 1000
sum_hyp = 0
start = time.time()
for i in range(N_graphs):
    ridx = random.randint(0, len(zinc))
    graph = zinc[ridx]
    adj = convert_zinc_to_nx(graph)
    hyp = hyperbolicity_sample(adj)
    sum_hyp += hyp
end = time.time()    
avg_hyp_zinc = sum_hyp / N_graphs    

print ("Average hyp for ZINC: ", avg_hyp_zinc)
print ("Time taken", end - start)

# dataset = 'pubmed'
# data_path = os.path.join(os.environ['DATAPATH'], dataset)
# # data = load_data_lp(dataset, use_feats=False, data_path=data_path)
# data = load_data_nc(dataset, use_feats=False, data_path=data_path, split_seed=47)
# graph = nx.from_scipy_sparse_matrix(data['adj_train'])

# print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
# hyp = hyperbolicity_sample(graph)
# print('Hyp: ', hyp)

