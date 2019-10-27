import os
import time
import numpy as np

import sklearn
from sklearn import metrics
import argparse
import torch.nn as nn
import torch
from torch.autograd import Variable
from sklearn.metrics import f1_score

from graphsage.unsupervised_models import UnsupervisedGraphSage
from graphsage.neigh_samplers import UniformNeighborSampler
import torch.nn.functional as F
from graphsage.dataset import Dataset
from graphsage.aggregators import MeanAggregator, MeanPoolAggregator, MaxPoolAggregator, LSTMAggregator
from graphsage.preps import NodeEmbeddingPrep

# Use in evaluate
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
#

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised graphSAGE")
    parser.add_argument('--prefix',           default="example_data/cora/graphsage/cora",  help="Data directory with prefix")
    parser.add_argument('--cuda',             default=False,           help="Run on cuda or not", type=bool)
    parser.add_argument('--model',            default='graphsage_mean',help='Model names. See README for possible values.')
    parser.add_argument('--multiclass',       default=False,           help='Whether use 1-hot labels or indices.', type=bool)
    parser.add_argument('--learning_rate',    default=0.01,            help='Initial learning rate.', type=float)
    parser.add_argument('--concat',           default=True,            help='Whether to concat', type=bool)
    parser.add_argument('--epochs',           default=10,              help='Number of epochs to train.', type=int)
    parser.add_argument('--max_degree',       default=25,              help='Maximum node degree.', type=int)
    parser.add_argument('--samples_1',        default=25,              help='Number of samples in layer 1', type=int)
    parser.add_argument('--samples_2',        default=10,              help='Number of samples in layer 2', type=int)
    parser.add_argument('--dim_1',            default=128,             help='Size of output dim (final is 2x this, if using concat)', type=int)
    parser.add_argument('--dim_2',            default=128,             help='Size of output dim (final is 2x this, if using concat)', type=int)
    parser.add_argument('--batch_size',       default=512,             help='Minibatch size.', type=int)
    parser.add_argument('--base_log_dir',     default='.',             help='Base directory for logging and saving embeddings')
    parser.add_argument('--print_every',      default=10,              help="How often to print training info.", type=int)
    parser.add_argument('--max_total_steps',  default=10**10,          help="Maximum total number of iterations", type=int)
    parser.add_argument('--validate_batch_size', default=256,          help="How many nodes per validation sample.", type=int)
    parser.add_argument('--neg_sample_size',  default=20,              help='Negative sample size', type=int)
    parser.add_argument('--identity_dim',     default=0,               help='Set to positive value to use identity embedding features of that dimension. Default 0.', type=int)
    parser.add_argument('--load_walks',        default=False,          help="Whether to load walk file.", type=bool)
    parser.add_argument('--num_walk',         default=50,              help="Number of walk from each node.", type=int)
    parser.add_argument('--walk_len',         default=5,               help="Length of each walk.", type=int)
    parser.add_argument('--seed',             default=123,             help="Random seed", type=int)
    parser.add_argument('--no_feature',       default=False,           help='whether to use features')

    return parser.parse_args()

def load_data(data_name, supervised=False, max_degree=25, multiclass=False, load_walks=True, num_walk=50, walk_len=5):
    dataset = Dataset()
    dataset.load_data(prefix=args.prefix, normalize=True, supervised=False, max_degree=max_degree, multiclass=multiclass, use_random_walks=True, load_walks=load_walks, num_walk=num_walk, walk_len=walk_len)
    return dataset

def log_dir(args):
    log_dir = args.base_log_dir + "/unsup"
    log_dir += "/{model:s}_{lr:0.6f}/".format(
            model=args.model,
            lr=args.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    print("Running regression...")
    dummy = MultiOutputClassifier(DummyClassifier())
    dummy.fit(train_embeds, train_labels)
    log = MultiOutputClassifier(SGDClassifier(loss="log", max_iter=5, tol=None), n_jobs=10)
    log.fit(train_embeds, train_labels)

    log_predict = log.predict(test_embeds)
    f1_sc = f1_score(test_labels, log_predict, average='micro')
    print("Average F1 score:", f1_sc)
    return f1_sc

def to_word2vec_format(val_embeddings, nodes, output_file_name, dim, pref=""):
    with open(output_file_name, 'w') as f_out:
        f_out.write("%s %s\n"%(len(nodes), dim))
        for i, node in enumerate(nodes):
            txt_vector = ["%s" % val_embeddings[i][j] for j in range(dim)]
            f_out.write("%s%s %s\n" % (pref, node, " ".join(txt_vector)))
        f_out.close()


def evaluate(model, dataset, args):
    if dataset.class_map == None:
        print("Currently not support evaluation without class map.")
        return -1

    # Get embeddings of all nodes
    embeddings = [] # embed of all nodes
    seen = set([])
    iter_num = 0
    while True:
        batch_nodes = dataset.nodes_ids[iter_num*args.validate_batch_size:(iter_num+1)*args.validate_batch_size]
        batch_nodes = torch.LongTensor([dataset.id_map[id] for id in batch_nodes])
        if args.cuda:
            batch_nodes = batch_nodes.cuda()
        if batch_nodes.shape[0] == 0: break
        outputs1, _, _ = model(batch_nodes, batch_nodes, mode="val")
        for i, node in enumerate(batch_nodes):
            if not node in seen:
                embeddings.append(outputs1[i,:].cpu().detach().numpy())
                # nodes.append(node.cpu().numpy().tolist())
                seen.add(node)
        iter_num += 1
    embeddings = np.vstack(embeddings)
    train_labels = np.array([dataset.class_map[id] for id in dataset.train_nodes_ids])
    test_labels = np.array([dataset.class_map[id] for id in dataset.test_nodes_ids])
    train_embeds = embeddings[[dataset.id_map[id] for id in dataset.train_nodes_ids]]
    test_embeds = embeddings[[dataset.id_map[id] for id in dataset.test_nodes_ids]]
    f1_sc = run_regression(train_embeds, train_labels, test_embeds, test_labels)
    return f1_sc


def train_(graphsage, train_edges, optimizer, epochs, batch_size = 256, cuda = False, args=None):
    avg_time = 0.0

    n_iters = len(train_edges)//batch_size
    #len(train_edges)%batch_size for case len%batch_size = 0
    if(len(train_edges) % batch_size > 0):
        n_iters = n_iters + 1
    total_steps = 0
    for epoch in range(epochs):
        print("Epoch {0}".format(epoch))
        np.random.shuffle(train_edges)
        for iter in range(n_iters):
            batch_edges = torch.LongTensor(train_edges[iter*batch_size:(iter+1)*batch_size])
            if cuda:
                batch_edges = batch_edges.cuda()

            t = time.time()
            optimizer.zero_grad()
            loss, outputs1, outputs2, neg_outputs = graphsage.loss(batch_edges[:,0], batch_edges[:,1])                
            loss.backward()
            nn.utils.clip_grad_value_(graphsage.parameters(), 5.0)
            optimizer.step()
            # print(graphsage.prep.embedding.weight)

            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % args.print_every == 0:
                mrr = graphsage.accuracy(outputs1, outputs2, neg_outputs)
                print("Iter:", '%03d' %iter,
                    "train_loss=", "{:.5f}".format(loss.item()),
                    "train_f1_mrr", "{:.5f}".format(mrr),
                    "time", "{:.5f}".format(avg_time)
                    )

            total_steps += 1
            if total_steps > args.max_total_steps:
                break
        if total_steps > args.max_total_steps:
            break
    return avg_time


def train(dataset, args):
    if args.no_feature:
        dataset.feats = None
    if dataset.feats is None:
        if args.identity_dim == 0:
            raise Exception("Must have a positive value for identity feature dimension if no input features given.")
        feat_dims = 0
        features = None
    else:
        feat_dims = dataset.feats.shape[1]
        features = torch.FloatTensor(dataset.feats)        
        if(args.cuda):
            features = features.cuda()

    if args.identity_dim != 0:
        feat_dims = feat_dims + args.identity_dim
    
    aggregator_cls = None

    if args.model == "graphsage_mean":
        aggregator_cls = MeanAggregator
    elif args.model == "graphsage_meanpool":
        aggregator_cls = MeanPoolAggregator
    elif args.model == "graphsage_maxpool":
        aggregator_cls = MaxPoolAggregator
    elif args.model == "graphsage_lstm":
        aggregator_cls = LSTMAggregator
    else:
        raise Exception("Unknown aggregator: ", args.model)
        
    if args.samples_2 != 0:
        agg1 = aggregator_cls(input_dim=feat_dims, output_dim=args.dim_1, activation=F.relu, concat=args.concat, dropout=0.0)
        agg2 = aggregator_cls(input_dim=agg1.output_dim, output_dim=args.dim_2, activation=False, concat=args.concat, dropout=0.0)
        agg_layers = [agg1, agg2]
        n_samples = [args.samples_1, args.samples_2]
    else:
        agg_layers = [aggregator_cls(input_dim=feat_dims, output_dim=args.dim_1, activation=False, concat=args.concat, dropout=0.0)]
        n_samples = [args.samples_1]

    # Transform adj from numpy array to torch tensor
    train_adj = torch.LongTensor(dataset.train_adj)
    adj = torch.LongTensor(dataset.adj)
    if args.cuda:
        train_adj = train_adj.cuda()
        adj = adj.cuda()

    train_edges = dataset.train_edges
    
    model = UnsupervisedGraphSage(
                                features = features,
                                train_adj = train_adj,
                                adj = adj,
                                train_deg = dataset.train_deg,
                                deg = dataset.deg,
                                agg_layers=agg_layers,
                                n_samples=n_samples,
                                sampler=UniformNeighborSampler(train_adj),
                                fc=False,
                                identity_dim=args.identity_dim,
                                neg_sample_size=args.neg_sample_size)
    if args.cuda:
        model = model.cuda()    

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.0)
    print("Start training")
    average_time = train_(model, train_edges, optimizer, args.epochs, args.batch_size, args.cuda, args)

    f1_sc = evaluate(model, dataset, args)
    print("F1 score = {0}".format(f1_sc))  

    return f1_sc, average_time

if __name__ == "__main__":
    args = parse_args()
    print(args)

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_data(args.prefix, supervised=False, max_degree = args.max_degree, multiclass=args.multiclass,
                    load_walks=args.load_walks, num_walk=args.num_walk, walk_len=args.walk_len)
    print("Start training....")
    f1_mics = []
    times = []
    for i in range(1):
        print("Training {0}".format(i))
        f1_mic, average_time = train(data, args)
        times.append(average_time)
        f1_mics.append(f1_mic)

    print("Final average validation F1 micro: ", np.mean(f1_mics))
    print("Final average batch time:{0}".format(np.mean(times)))



