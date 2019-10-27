from collections import defaultdict
import argparse
import numpy as np
import json
import os
import networkx as nx
from networkx.readwrite import json_graph

def parse_args():
    parser = argparse.ArgumentParser(description="Generate graphsage format from edgelist")        
    parser.add_argument('--out_dir', default=None, help="Output directory")
    parser.add_argument('--prefix', default="karate", help="seed")
    return parser.parse_args()

def edgelist_to_graphsage(dataset, dir):    
    edgelist_dir = dir + "/edgelist/" + dataset + ".edgelist"
    G = nx.read_edgelist(edgelist_dir)
    print(nx.info(G))
    num_nodes = len(G.nodes())
    rand_indices = np.random.permutation(num_nodes)
    train = rand_indices[:int(num_nodes * 0.81)]
    val = rand_indices[int(num_nodes * 0.81):int(num_nodes * 0.9)]
    test = rand_indices[int(num_nodes * 0.9):]

    id_map = {}
    for i, node in enumerate(G.nodes()):
        id_map[str(node)] = i

    res = json_graph.node_link_data(G)    
    res['nodes'] = [
        {
            'id': node['id'],
            'val': id_map[str(node['id'])] in val,
            'test': id_map[str(node['id'])] in test
        }
        for node in res['nodes']]
                    
    res['links'] = [
        {
            'source': link['source'],
            'target': link['target']
        }
        for link in res['links']]

    if not os.path.exists(dir + "/graphsage/"):
        os.makedirs(dir + "/graphsage/")
                
    with open(dir + "/graphsage/" + dataset + "-G.json", 'w') as outfile:
        json.dump(res, outfile)
    with open(dir + "/graphsage/" + dataset + "-id_map.json", 'w') as outfile:
        json.dump(id_map, outfile)
        
    print("GraphSAGE format stored in {0}".format(dir + "/graphsage/"))
    print("----------------------------------------------------------")

if __name__ == "__main__":
    args = parse_args()
    datadir = args.out_dir
    dataset = args.prefix
    edgelist_to_graphsage(dataset, datadir)



