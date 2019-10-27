import numpy as np
from collections import defaultdict
import json
from sklearn.preprocessing import OneHotEncoder
import os
file_dir = os.path.dirname(os.path.realpath(__file__))

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open(file_dir + "/../example_data/pubmed/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open(file_dir + "/../example_data/pubmed/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists, node_map

def toGraphsageFormat(feats, labels, adj_lists, id_map):
  # transform labels to 1-hot vector
  enc = OneHotEncoder(dtype=np.int64)
  enc.fit(labels)
  labels = enc.transform(labels).toarray()

  G = {
    "directed": False,
    "multigraph": False,
    "nodes": [], # id: int, feature: [], label: [], val: Bool, test: Bool
    "links": [] # source: int, target: int, test_removed: Bool, train_removed: Bool
  }
  class_map = {}

  num_nodes = feats.shape[0]
  num_train = int(num_nodes*0.81)
  num_val = int(num_nodes*0.09)

  reversed_idmap = {v:k for k, v in id_map.items()}

  for idx in range(num_nodes): # iterate through number of nodes
    node = {
      "id": reversed_idmap[idx],
      "feature": feats[idx].tolist(),
      "label": labels[idx].tolist()
    }
    if idx < num_train:
      node["val"] = False
      node["test"] = False
    elif idx < num_train + num_val:
      node["val"] = True
      node["test"] = False
    else:
      node["val"] = False
      node["test"] = True

    G["nodes"].append(node)
    class_map[reversed_idmap[idx]] = labels[idx].tolist()
    for target in adj_lists[idx]:
      if target < idx: continue
      link = {
        "source": idx,
        "target": target
      }
      G["links"].append(link)

  print("Export pubmed-G.json")
  with open(file_dir + "/../example_data/pubmed/graphsage/pubmed-G.json", "w+") as file:
    json.dump(G, file)
  print("Export pubmed-class_map.json")
  with open(file_dir + "/../example_data/pubmed/graphsage/pubmed-class_map.json", "w+") as file:
    json.dump(class_map, file)
  print("Export pubmed-feats.npy")
  np.save(file_dir + "/../example_data/pubmed/graphsage/pubmed-feats.npy", feats)
  print("Export pubmed-id_map.json")
  with open(file_dir + "/../example_data/pubmed/graphsage/pubmed-id_map.json", "w+") as file:
    json.dump(id_map, file)

if __name__ == "__main__":
  feats, labels, adj_lists, id_map = load_pubmed()
  toGraphsageFormat(feats, labels, adj_lists, id_map)