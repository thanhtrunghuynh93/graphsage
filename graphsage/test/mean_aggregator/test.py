"""
  Copy code to end of file graphsage/aggregators.py and then run.
"""

import numpy as np

if __name__ == '__main__':
    # file self_vecs.npy, neigh_vecs.npy contains the inputs self_vecs, neigh_vecs from MeanAggregator GraphSAGE Tensorflow
    # file out.npy: return from MeanAggregator Tensorflow
    self_vecs = np.load("graphsage/test/mean_aggregator/self_vecs.npy")
    neigh_vecs = np.load("graphsage/test/mean_aggregator/neigh_vecs.npy")
    out = np.load("graphsage/test/mean_aggregator/out.npy")

    self_vecs = nn.Parameter(torch.FloatTensor(self_vecs))
    neigh_vecs = nn.Parameter(torch.FloatTensor(neigh_vecs))
    agg = MeanAggregatorNew(256, 128, concat=True, act=lambda x: x, cuda=False)
    out2 = agg(self_vecs, neigh_vecs)

    print("The values of out and out2 can be different due to the differentation of weights.")
    print("Check shape and flow only.")
    print("Shape out: {} | Shape out2: {}".format(out.shape, out2.shape))
    if out.shape == out2.detach().numpy().shape:
        print("Correct!")
    else:
        print("Not match.")