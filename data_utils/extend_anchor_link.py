from __future__ import print_function, division
from copy import deepcopy
import numpy as np
import random
import json
import sys
import os
import argparse
from shutil import copyfile

import networkx as nx
from networkx.readwrite import json_graph
import pdb

"""
input: source_net, target_net, alpha_t
"""

def parse_args():
    source_path = '$HOME/dataspace/graph/pale_facebook/random_clone/sourceclone,alpha_c=0.9,alpha_s=0.9'
    target_path = '$HOME/dataspace/graph/pale_facebook/random_clone/targetclone,alpha_c=0.9,alpha_s=0.9'
    source_out = '$HOME/dataspace/graph/pale_facebook/random_clone/extend1_c0.9s0.9t0.2'
    target_out = '$HOME/dataspace/graph/pale_facebook/random_clone/extend2_c0.9s0.9t0.2'
    parser = argparse.ArgumentParser(description="Extend netword")
    parser.add_argument('--input1', default=source_path)
    parser.add_argument('--input2', default=target_path)
    parser.add_argument('--output1', default=source_out)
    parser.add_argument('--output2', default=target_out)
    parser.add_argument('--prefix', default='pale_facebook')
    parser.add_argument('--alpha_t', default=0.03, type=float)
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    return parser.parse_args()

def construct_data(G1, G2, original_idmap):
    n_nodes = len(G1.nodes())
    n_train_nodes = int(n_nodes*args.alpha_t)
    train_node_ids = list(np.random.choice(G1.nodes(), n_train_nodes, replace=False))
    for ele in train_node_ids:
        G1.node[ele]['mapping_train'] = True
        G2.node[ele]['mapping_train'] = True
    test_node_ids = [node for node in G1.nodes() if node not in train_node_ids]
    for ele in test_node_ids:
        G1.node[ele]['mapping_train'] = False
        G2.node[ele]['mapping_train'] = False
    target_idmap = deepcopy(original_idmap)

    # shuffle idx 
    node_idxs = [target_idmap[id_] for id_ in G1.nodes()]
    random.shuffle(node_idxs)


    for i, id_ in enumerate(G1.nodes()):
        target_idmap[id_] = node_idxs[i]

    # extending...
    for node in train_node_ids:
        n1 = G1.neighbors(node)
        n2 = G2.neighbors(node)
        # print(type(n1))
        observed_n1 = [n for n in n1 if n in train_node_ids]
        observed_n2 = [n for n in n2 if n in train_node_ids]
        observed_n = observed_n1 + observed_n2
        for n in observed_n:
            G1.add_edge(node, n)
            G2.add_edge(node, n)
    
    return G1, G2, target_idmap


def main(args):
    args.input1 += "/" + args.prefix
    args.input2 += "/" + args.prefix
    G_data1 = json.load(open(args.input1 + "-G.json"))
    G_data2 = json.load(open(args.input2 + "-G.json"))
    original_idmap = json.load(open(args.input1 + "-id_map.json"))
    G1 = json_graph.node_link_graph(G_data1)
    G2 = json_graph.node_link_graph(G_data2)
    # print(nx.info(G1))
    H1 = G1.copy()
    H2 = G2.copy()
    G1_new, G2_new, G2_idmap_new = construct_data(H1, H2, original_idmap)

    print("This is G1 ori info: ")
    print(nx.info(G1), "\n")

    print("This is G1 info: ")
    print(nx.info(G1_new), "\n")

    print("This is G2 ori info: ")
    print(nx.info(G2), "\n")

    print("This is G2 info: ")
    print(nx.info(G2_new), "\n")


    data1 = json_graph.node_link_data(G1_new)
    data2 = json_graph.node_link_data(G2_new)
    s1 = json.dumps(data1, indent=4, sort_keys=True)
    s2 = json.dumps(data2, indent=4, sort_keys=True)

    edgelist_dir1 = args.output1 + '/' + args.prefix + ".edgelist"
    edgelist_dir2 = args.output2 + '/' + args.prefix + ".edgelist"

    if not os.path.isdir(args.output1): os.makedirs(args.output1)
    if not os.path.isdir(args.output2): os.makedirs(args.output2)

    nx.write_edgelist(G1_new, path = edgelist_dir1 , delimiter=" ", data=['weight'])
    nx.write_edgelist(G2_new, path = edgelist_dir2 , delimiter=" ", data=['weight'])

    args.output1 += "/" + args.prefix
    args.output2 += "/" + args.prefix

    with open(args.output1 + "-G.json", 'w') as f:
        f.write(s1)
        f.close()

    with open(args.output2 + "-G.json", 'w') as f:
        f.write(s2)
        f.close()

    copyfile(args.input1 + "-id_map.json", args.output1 + "-id_map.json")
    with open(args.output2 + "-id_map.json", 'w') as f:
        json.dump(G2_idmap_new, f)
        f.close()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)