from __future__ import print_function, division
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
from collections import defaultdict
import random
import scipy.io as sio
from scipy.sparse import coo_matrix
from scipy import sparse
import argparse
import math
import sys
import csv
import pdb
import os

# Author: TrungHT
# Created time: 18/6/2018

class MatLinkagePreprocessor:
    
    def __init__(self, mat_file_dir, dataset_name_1, dataset_name_2, output_dir):
        self.mat_file_dir = mat_file_dir
        self.dataset_name_1 = dataset_name_1
        self.dataset_name_2 = dataset_name_2
        self.output_dir = output_dir


    def process(self):

        dict = sio.loadmat(self.mat_file_dir)
        print("-------------Processing file at {0} and generating to GraphSAGE format and edgelist---------------".format(self.mat_file_dir))
        print("File specs:")

        print(dict.keys())

        key_list = []
        key_list.extend([self.dataset_name_1, self.dataset_name_1 + "_node_label", self.dataset_name_1 + "_edge_label"])
        key_list.extend([self.dataset_name_2, self.dataset_name_2 + "_node_label", self.dataset_name_2 + "_edge_label"])
        

        if "gndtruth" in dict.keys():
            key_list.extend(["gndtruth"])
            self.storeGroundTruth(dict["gndtruth"])
        if "ground_truth" in dict.keys():
            key_list.extend(["ground_truth"])
            self.storeGroundTruth(dict["ground_truth"])
        if "H" in dict.keys():
            print(dict['H'].shape)
            key_list.extend(["H"])  
            self.storeH(dict["H"])     
        
        for key in key_list:
            print("{0} : {1} {2}".format(key, type(dict[key]), dict[key].shape))  
        print("---------------------------------------------------------------------------------")

        self.processDataset(self.dataset_name_1, 
                            dict[self.dataset_name_1], 
                            dict[self.dataset_name_1 + "_node_label"], 
                            dict[self.dataset_name_1 + "_edge_label"])

        self.processDataset(self.dataset_name_2, 
                            dict[self.dataset_name_2], 
                            dict[self.dataset_name_2 + "_node_label"], 
                            dict[self.dataset_name_2 + "_edge_label"])
        
    def processDataset(self, dataset_name, adjacency_matrix, node_label, edge_label):
        
        edge_labels = []
        for ele in edge_label[0]:
            edge_labels.append(ele.todense())
        dense_edge_label = np.array(edge_labels)

        print("Processing dataset {0}".format(dataset_name))
        dir = "{0}/{1}".format(self.output_dir,dataset_name)
        if not os.path.exists(dir + "/edgelist/"):
            os.makedirs(dir + "/edgelist/")
        dense_node_label = node_label.todense() # label can be 0, 1 or 2 | to one hot vector

        sources, targets = adjacency_matrix.nonzero() # source 0 -> 1117


        edgelist = zip(sources.tolist(), targets.tolist())
        G = nx.Graph(edgelist)
        edgelist_dir = dir + "/edgelist/" + dataset_name + ".edgelist"
        nx.write_edgelist(G, path = edgelist_dir , delimiter=" ", data=['weight'])

        print(nx.info(G))        
        print("Edgelist stored in {0}".format(edgelist_dir))

        # G = nx.read_edgelist(edgelist_dir)

        num_nodes = len(G.nodes())

        rand_indices = np.random.permutation(num_nodes)
        train = rand_indices[:int(num_nodes * 0.81)]

        val = rand_indices[int(num_nodes * 0.81):int(num_nodes * 0.9)]
        test = rand_indices[int(num_nodes * 0.9):]

        id_map = {}
        for i, node in enumerate(G.nodes()):
            id_map[str(node)] = i

        # print(G.edges())

        res = json_graph.node_link_data(G)

        res['nodes'] = [
            {
                'id': str(node['id']),
                'feature': np.squeeze(np.asarray(dense_node_label[int(node['id'])])).tolist(),
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

        dir += "/graphsage/"
        if not os.path.exists(dir):
            os.makedirs(dir)
                       
        with open(dir + dataset_name + "-G.json", 'w') as outfile:
            json.dump(res, outfile)       
        
        with open(dir + dataset_name + "-id_map.json", 'w') as outfile:
            json.dump(id_map, outfile)
        
        feats = np.zeros((dense_node_label.shape[0], dense_node_label.shape[1]))
        edge_feats = np.zeros_like(dense_edge_label)
        for id in id_map.keys():
            idx = id_map[id]
            feats[idx] = dense_node_label[int(id)]
            for id_ in id_map.keys():
                idx_ = id_map[id_]
                for l in range(edge_feats.shape[0]):
                    edge_feats[int(l), int(idx), int(idx_)] = dense_edge_label[int(l), int(id), int(id_)]

        np.save(dir + dataset_name + "-feats.npy",  feats)
        np.save(dir + dataset_name + "-edge_feats.npy", edge_feats)
        print(dir + dataset_name)

        print("GraphSAGE format stored in {0}".format(dir))
        print("----------------------------------------------------------")
        return G


    def storeH(self, H):
        try:
            H = H.todense()
        except:
            pass
        np.save(self.output_dir + '/H.npy', H)
        print("Prior H have been saved to: ", self.output_dir + '/H.npy')

    def storeGroundTruth(self, groundTruth):
        dir = self.output_dir + "/dictionaries/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        file = open(dir + "groundtruth","w") 
        
        for edge in groundTruth:
            #Matlab index count from 1
            file.write("{0} {1}\n".format(int(edge[0]) - 1, int(edge[1]) - 1)) 
        file.close() 

def parse_args():
    parser = argparse.ArgumentParser(description="Convert mat linkage data to dataset's edgelist and GraphSAGE format.")
    parser.add_argument('--input', default="data.mat", help="Input data directory")
    parser.add_argument('--dataset1', default="", help="Name of the dataset 1")
    parser.add_argument('--dataset2', default="", help="Name of the dataset 2")
    parser.add_argument('--output', default="", help="Input data directory")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    processor = MatLinkagePreprocessor(args.input, args.dataset1, args.dataset2, args.output)
    processor.process()

    
        
