import tensorflow as tf
import numpy as np
import pdb

import torch
labels = np.load('data/labels.npy')
degrees = np.load('data/degrees.npy')


neg_samples, true_count, sampled_count = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=20,
            unique=False,
            range_max=len(degrees),
            distortion=0.75,
            unigrams=degrees.tolist())

samples_tf = []
trues = []
sampleds = []
print("Run TF")
with tf.Session() as sess:
  for i in range(100000):
    outs = sess.run([neg_samples, true_count, sampled_count])
    samples_tf.append(outs[0])
    # trues.append(outs[1])
    # sampleds.append(outs[2])
print("Run pytorch")
def fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, distortion, unigrams):
  weights = unigrams**distortion
  prob = weights/weights.sum()
  sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
  return sampled

samples_py = []
for i in range(100000):
  sample = fixed_unigram_candidate_sampler(
              true_classes=labels.flatten(),
              num_true=1,
              num_sampled=20,
              unique=False,
              range_max=len(degrees),
              distortion=0.75,
              unigrams=degrees)
  samples_py.append(sample)

# check num true in sample
counts_true_tf = []
counts_true_py = []
for sample in samples_tf:
  labels_flatten = labels.flatten()
  count_true = sum([1 for s in sample if s in labels_flatten])
  counts_true_tf.append(count_true)
for sample in samples_py:
  labels_flatten = labels.flatten()
  count_true = sum([1 for s in sample if s in labels_flatten])
  counts_true_py.append(count_true)

print("Average TF: {}".format(sum(counts_true_tf)/len(counts_true_tf)))
print("Average Pytorch: {}".format(sum(counts_true_py)/len(counts_true_py)))


# check true_count
# print(labels.shape)
# for idx, true in enumerate(trues):
#   if idx == 0: continue
#   correct = true == trues[0]
#   print(correct.sum())

# check sampleds count
# for idx, sampled in enumerate(sampleds):
#   if idx == 0: continue
#   corr = sampled == sampleds[0]
#   print(corr.sum())

# check probability
# weights = degrees**0.75
# prob = weights/weights.sum()
# true_prob = prob[labels.flatten()]
# print(true_prob[:10])

# print(trues[0][:10, 0])

# print(trues[0].sum())
# print(sampleds[0].sum())


