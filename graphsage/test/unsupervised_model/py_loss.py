import torch

def affinity(inputs1, inputs2):
    result = torch.sum(inputs1 * inputs2, dim=1)
    return result

def neg_cost(inputs1, neg_samples):
    neg_aff = inputs1.mm(neg_samples.t())
    return neg_aff

def sigmoid_cross_entropy_with_logits(labels, logits):
    sig_aff = torch.sigmoid(logits)
    loss = labels * -torch.log(sig_aff) + (1 - labels) * -torch.log(1 - sig_aff)    
    return loss

def loss(inputs1, inputs2, neg_samples):
    cuda = inputs1.is_cuda
    true_aff = affinity(inputs1, inputs2)
    neg_aff = neg_cost(inputs1, neg_samples)
    true_labels = torch.ones(true_aff.shape)
    if cuda:
        true_labels = true_labels.cuda()
    true_xent = sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
    neg_labels = torch.zeros(neg_aff.shape)
    if cuda:
        neg_labels = neg_labels.cuda()
    neg_xent = sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
    loss = true_xent.sum() + 1.0 * neg_xent.sum()
    return loss

if __name__ == "__main__":
    inputs1 = [[-0.0303, -0.3279, -0.2467, -0.9114], [-0.0303, -0.3279, -0.2467, -0.9114]]
    inputs2 = [[ 0.3110, -0.5717, -0.2602, -0.7133], [ 0.3110, -0.5717, -0.2602, -0.7133]]
    neg_samples = [[-0.1305, -0.6129, -0.1885, -0.7562],
        [ 0.2049, -0.0324, -0.4317, -0.8778],
        [ 0.1177,  0.0196, -0.3731, -0.9201],
        [ 0.3110, -0.5718, -0.2370, -0.7212],
        [-0.0292, -0.9820,  0.0036, -0.1865],
        [ 0.0000,  0.0000, -0.2413, -0.9705],
        [-0.1655, -0.0263, -0.3147, -0.9343],
        [-0.1488, -0.0225, -0.4417, -0.8844],
        [-0.3110, -0.0570, -0.3290, -0.8898],
        [-0.1904, -0.2189, -0.4283, -0.8558],
        [ 0.2833, -0.3374, -0.5130, -0.7367],
        [ 0.1452,  0.0212, -0.2635, -0.9534],
        [-0.0629, -0.0426, -0.2735, -0.9589],
        [-0.5989, -0.5239, -0.3061, -0.5227],
        [ 0.0000,  0.0000, -0.3201, -0.9474],
        [-0.0918, -0.0106, -0.2147, -0.9723],
        [ 0.6054, -0.7313,  0.0758, -0.3050],
        [ 0.5552, -0.0591, -0.2455, -0.7925],
        [ 0.1858, -0.4874, -0.1530, -0.8394],
        [-0.4677, -0.4390, -0.2287, -0.7323]]
    print(loss(torch.FloatTensor(inputs1), torch.FloatTensor(inputs2), torch.FloatTensor(neg_samples)))