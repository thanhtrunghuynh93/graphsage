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

from graphsage.supervised_models import SupervisedGraphSage
from graphsage.neigh_samplers import UniformNeighborSampler
import torch.nn.functional as F
from graphsage.dataset import Dataset
from graphsage.aggregators import MeanAggregator, MeanPoolAggregator, MaxPoolAggregator, LSTMAggregator, SumAggregator

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised graphSAGE")
    parser.add_argument('--prefix',           default="example_data/cora/graphsage/cora",  help="Data directory with prefix")
    parser.add_argument('--cuda',             default=False,           help="Run on cuda or not", type=bool)
    parser.add_argument('--model',            default='graphsage_mean',help='Model names. See README for possible values.')
    parser.add_argument('--multiclass',       default=False,           help='Whether use 1-hot labels or indices.', type=bool)
    parser.add_argument('--learning_rate',    default=0.01,            help='Initial learning rate.', type=float)
    parser.add_argument('--concat',           default=True,            help='whether to concat', type=bool)
    parser.add_argument('--epochs',           default=10,              help='Number of epochs to train.', type=int)
    parser.add_argument('--max_degree',       default=25,              help='Maximum node degree.', type=int)
    parser.add_argument('--samples_1',        default=10,              help='Number of samples in layer 1', type=int)
    parser.add_argument('--samples_2',        default=25,              help='Number of samples in layer 2', type=int)
    parser.add_argument('--dim_1',            default=128,             help='Size of output dim (final is 2x this, if using concat)', type=int)
    parser.add_argument('--dim_2',            default=128,             help='Size of output dim (final is 2x this, if using concat)', type=int)
    parser.add_argument('--batch_size',       default=256,             help='Minibatch size.', type=int)
    parser.add_argument('--base_log_dir',     default='.',             help='Base directory for logging and saving embeddings')
    parser.add_argument('--print_every',      default=10,              help="How often to print training info.", type=int)
    parser.add_argument('--max_total_steps',  default=10**10,          help="Maximum total number of iterations", type=int)
    parser.add_argument('--validate_iter',    default=5000,            help="How often to run a validation minibatch.", type=int)
    parser.add_argument('--validate_batch_size', default=256,          help="How many nodes per validation sample.", type=int)
    parser.add_argument('--identity_dim',     default=0,               help='Set to positive value to use node_embedding_prep. Default 0.', type=int)
    parser.add_argument('--seed',             default=123,             help="Random seed", type=int)
    parser.add_argument('--no_feature',       default=False,          help='whether to use features')


    return parser.parse_args()

def load_data(data_name, supervised=True, max_degree=25, multiclass=False, load_adj_dir = None):
    dataset = Dataset()
    dataset.load_data(prefix = args.prefix, normalize=True, supervised=True, max_degree=max_degree, multiclass=multiclass, load_adj_dir = load_adj_dir)
    return dataset

def log_dir(args):
    log_dir = args.base_log_dir + "/sup-" + args.prefix.split("/")[-4]
    log_dir += "/{model:s}_{lr:0.6f}/".format(
            model=args.model,
            lr=args.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def calc_f1(y_true, y_pred):
    if not args.multiclass:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

    return f1_score(y_true, y_pred, average="micro"), f1_score(y_true, y_pred, average='macro')

def evaluate(graphsage, batch_nodes, labels, args, mode="val"):   
    batch_nodes = batch_nodes.cpu()
    t = time.time()    
    output = graphsage.forward(batch_nodes, mode=mode)
    if args.multiclass:
        f1_mic, f1_mac = calc_f1(labels[batch_nodes], output.data.cpu().numpy())
    else:
        f1_mic = f1_score(labels[batch_nodes], output.data.cpu().numpy().argmax(axis=1), average="micro")
        f1_mac = f1_score(labels[batch_nodes], output.data.cpu().numpy().argmax(axis=1), average='macro')
    return f1_mic, f1_mac, time.time() - t

def to_word2vec_format(val_embeddings, nodes, output_file_name, dim, pref=""):
    with open(output_file_name, 'w') as f_out:
        f_out.write("%s %s\n"%(len(nodes), dim))
        for i, node in enumerate(nodes):
            txt_vector = ["%s" % val_embeddings[i][j] for j in range(dim)]
            f_out.write("%s%s %s\n" % (pref, node, " ".join(txt_vector)))
        f_out.close()

def train_(graphsage, train_nodes, val_nodes, labels, optimizer, epochs, batch_size = 256, multiclass = False, cuda = False, args=None):
    avg_time = 0.0
    n_iters = len(train_nodes)//batch_size
    #len(train_edges)%batch_size for case len%batch_size = 0
    if(len(train_nodes) % batch_size > 0):
        n_iters = n_iters + 1
    total_steps = 0
    for epoch in range(epochs):
        print("Epoch {0}".format(epoch))
        np.random.shuffle(train_nodes)
        for iter in range(n_iters):

            batch_nodes = torch.LongTensor(train_nodes[iter*batch_size:(iter+1)*batch_size])
            if cuda:
                batch_nodes = batch_nodes.cuda()

            t = time.time()
            optimizer.zero_grad()
            if multiclass:
                _labels = torch.FloatTensor(labels)[batch_nodes]
            else:
                _labels = torch.LongTensor(labels)[batch_nodes]
            if cuda:
                _labels = _labels.cuda()

            loss = graphsage.loss(batch_nodes, _labels)
            loss.backward()
            optimizer.step()

            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if iter % args.validate_iter == 0:
                np.random.shuffle(val_nodes)
                val_to_feed = torch.LongTensor(val_nodes[:args.validate_batch_size])
                if cuda:
                    val_to_feed = val_to_feed.cuda()
                val_f1_mic, val_f1_mac, _ = evaluate(graphsage, val_to_feed, labels, args)

            if total_steps % args.print_every == 0:
                train_f1_mic, train_f1_mac, _ = evaluate(graphsage, batch_nodes, labels, args)
                print("Iter:", '%03d' %iter,
                      "train_loss=", "{:.5f}".format(loss.item()),
                      "train_f1_mic", "{:.5f}".format(train_f1_mic),
                      "train_f1_mac", "{:.5f}".format(train_f1_mac),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                      "time", "{:.5f}".format(avg_time)
                      )

            total_steps += 1
            if total_steps > args.max_total_steps:
                break
        if total_steps > args.max_total_steps:
            break
    return avg_time


def train(dataset, args):
    num_classes = len(list(dataset.class_map.values())[0])
    if args.no_feature:
        dataset.feats = None
    if dataset.feats is None:
        assert args.identity_dim > 0, "if feats is None, requires identity_dim > 0"
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
    elif args.model == "graphsage_sum":
        aggregator_cls = SumAggregator
    elif args.model == "graphsage_meanpool":
        aggregator_cls = MeanPoolAggregator
    elif args.model == "graphsage_maxpool":
        aggregator_cls = MaxPoolAggregator
    elif args.model == "graphsage_lstm":
        aggregator_cls = LSTMAggregator
    else:
        raise Exception("Unknown aggregator: ", args.model)

    if args.samples_2 != 0:
        print("Using 2 aggregator layers")
        agg1 = aggregator_cls(input_dim=feat_dims, output_dim=args.dim_1, activation=F.relu, concat=args.concat, dropout=0.0)
        agg2 = aggregator_cls(input_dim=agg1.output_dim, output_dim=args.dim_2, activation=False, concat=args.concat, dropout=0.0)
        agg_layers = [agg1, agg2]
        n_samples = [args.samples_1, args.samples_2]
    else:
        print("Using 1 aggregator layers")
        agg_layers = [aggregator_cls(input_dim=feat_dims, output_dim=args.dim_1, activation=False, concat=args.concat, dropout=0.0)]
        n_samples = [args.samples_1]

    # Transform adj from numpy array to torch tensor
    train_adj = torch.LongTensor(dataset.train_adj)
    adj = torch.LongTensor(dataset.adj)
    if args.cuda:
        train_adj = train_adj.cuda()
        adj = adj.cuda()

    train_nodes = dataset.train_nodes
    val_nodes = dataset.val_nodes   

    model = SupervisedGraphSage(
                                features=features,
                                train_adj = train_adj,
                                adj = adj,
                                train_deg = dataset.train_deg,
                                deg = dataset.deg,
                                agg_layers=agg_layers,
                                n_samples=n_samples,
                                sampler=UniformNeighborSampler(train_adj),
                                fc=nn.Linear(agg_layers[-1].output_dim, num_classes, bias=True),
                                multiclass=args.multiclass,
                                identity_dim=args.identity_dim
                                )
    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.0)    
    # optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=args.learning_rate)
    average_time = train_(model, train_nodes, val_nodes, dataset.labels, optimizer, args.epochs, args.batch_size, args.multiclass, args.cuda, args)


    val = torch.LongTensor(val_nodes)
    if args.cuda:
        val = val.cuda()
    val_f1_mic, val_f1_mac, val_time = evaluate(model, val, dataset.labels, args)
    print("Validation f1 micro: ", val_f1_mic, " f1 macro: ", val_f1_mac)
    print("Average batch time:{0}".format(average_time))

    # with open(log_dir(args) + "val_stats.txt", "w") as fp:
    #     fp.write("f1_micro={:.5f} f1_macro={:.5f}".format(val_f1_mic, val_f1_mac))


    return val_f1_mic, val_f1_mac, average_time

if __name__ == "__main__":

    args = parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_data(args.prefix, supervised=True, max_degree = args.max_degree, multiclass=args.multiclass)
    print("Start training....")
    f1_mics = []
    f1_macs = []
    times = []
    for i in range(1):
        print("Training {0}".format(i))
        f1_mic, f1_mac, average_time = train(data, args)
        f1_mics.append(f1_mic)
        f1_macs.append(f1_mac)
        times.append(average_time)

    print("Final average F1 micro: ", np.mean(f1_mics), " F1 macro: ", np.mean(f1_macs))
    print("Final average batch time:{0}".format(np.mean(times)))
