from collections import defaultdict
import numpy as np
import json


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("../example_data/cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = [float(s) for s in info[1:-1]] 
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
    adj_lists = defaultdict(set)
    with open("../example_data/cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists, node_map


def to_graphsage_format(feat_data, labels, adj_lists, node_map):
    np.save('../example_data/cora/graphsage/cora-feats', feat_data)
    G = {'directed': False, 'graph': {'name': 'disjoint_union( , )'}}
    nodes = []
    class_map = {}
    num_nodes = feat_data.shape[0]
    num_train = int(num_nodes*0.81)
    num_val = int(num_nodes*0.09)

    real_ids = list(node_map.keys())

    for i in range(len(feat_data)):
        node_i = {}
        one_hot_label = [0]*7
        one_hot_label[labels[i][0]] = 1
        node_i['id'] = real_ids[i]
        
        if i < num_train:
            node_i['test'] = False
            node_i['val'] = False
        elif i >= num_train + num_val:
            node_i['test'] = True
            node_i['val'] = False
        else:
            node_i['test'] = False
            node_i['val'] = True

        node_i['feature'] = list(feat_data[i])
        node_i['labels'] = one_hot_label

        class_map[real_ids[i]] = one_hot_label
        nodes.append(node_i)
    G['nodes'] = nodes

    links = []
    for key in adj_lists:
        links += [{'source': key, 'target': list(adj_lists[key])[i]} for i in range(len(list(adj_lists[key]))) if list(adj_lists[key])[i] >= key]
    G['links'] = links
    G['multigraph'] = False
    with open('../example_data/cora/graphsage/cora-G.json', 'w') as outfile:
        json.dump(G, outfile)
    with open('../example_data/cora/graphsage/cora-id_map.json', 'w') as outfile:
        json.dump(node_map, outfile)
    with open('../example_data/cora/graphsage/cora-class_map.json', 'w') as outfile:
        json.dump(class_map, outfile)



if __name__ == "__main__":
    feat_data, labels, adj_lists, node_map = load_cora()
    to_graphsage_format(feat_data, labels, adj_lists, node_map)
