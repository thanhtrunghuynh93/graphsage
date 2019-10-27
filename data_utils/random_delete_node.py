from __future__ import print_function, division
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

def remove_node(G, p_remove, num_remove=10):
    '''
    for each node,
    remove with prob p
    operates on G in-place
    '''
    rem_nodes = []
    count_rm = 0
    for node in G.nodes():
        # probabilistically remove a random node
        if count_rm <= num_remove: # only try if
            if random.random() < p_remove:
                G.remove_node(node)
                rem_nodes.append(node)
                if count_rm % 1000 == 0:
                    print("\t{0}-th node removed:\t {1}".format(count_rm, node))
                count_rm += 1

        if count_rm > num_remove:
            break
    #Remove isolated node
    for node in G.nodes():
        if G.degree(node) == 0:
            G.remove_node(node)
            rem_nodes.append(node)
            count_rm += 1

    print("Remove total {0} nodes".format(count_rm))
    return rem_nodes

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly remove edges and generate dict.")
    parser.add_argument('--input', default=None, help='Path to load data')
    parser.add_argument('--output', default=None, help='Path to save data')
    parser.add_argument('--prefix', default=None, help='Dataset prefix')
    parser.add_argument('--ratio', type=float, default=0.2, help='Probability of remove nodes')
    parser.add_argument('--seed', type=int, default=121, help='Random seed')
    return parser.parse_args()

def main(args):
    args.input += "/" + args.prefix
    G_data = json.load(open(args.input + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    print(nx.info(G))

    H = G.copy()
    n = len(G.nodes())
    rem_nodes = remove_node(H, args.ratio, int(args.ratio * n))

    data = json_graph.node_link_data(H)
    s = json.dumps(data,  indent=4, sort_keys=True)
    print(nx.info(H))
    

    args.output += "/del,p={0}".format(args.ratio)
    if not os.path.isdir(args.output):
        os.makedirs(args.output+'/edgelist')
        os.makedirs(args.output+'/graphsage')
        os.makedirs(args.output+'/dictionaries')

    with open(args.output + "/dictionaries/groundtruth", 'w') as f:
        for node in G.nodes():
            if node not in rem_nodes:
                f.write("{0} {1}\n".format(node, node))
        f.close()

    edgelist_dir = args.output + "/edgelist/" + args.prefix + ".edgelist"
    nx.write_edgelist(H, path = edgelist_dir , delimiter=" ", data=['weight'])

    args.output += "/graphsage/" + args.prefix
    with open(args.output + "-G.json", 'w') as f:
        f.write(s)
        f.close()

    copyfile(args.input + "-id_map.json", args.output + "-id_map.json")
    if os.path.exists(args.input + "-class_map.json"):
        copyfile(args.input + "-class_map.json", args.output + "-class_map.json")
    if os.path.exists(args.input + "-feats.npy"):
        copyfile(args.input + "-feats.npy", args.output + "-feats.npy")
    return

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)