import tensorflow as tf

def affinity(inputs1, inputs2):
    result = tf.reduce_sum(inputs1 * inputs2, axis=1)
    return result

def neg_cost(inputs1, neg_samples, hard_neg_samples=None):
    neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))
    return neg_aff

def accuracy(outputs1, outputs2, neg_outputs):
    batch_size = outputs1.shape[0]
    neg_sample_size = neg_outputs.shape[0]
    # shape: [batch_size]
    aff = affinity(outputs1, outputs2)
    # shape : [batch_size x num_neg_samples]
    neg_aff = neg_cost(outputs1, neg_outputs)
    neg_aff = tf.reshape(neg_aff, [batch_size, neg_sample_size])
    _aff = tf.expand_dims(aff, axis=1)
    aff_all = tf.concat(axis=1, values=[neg_aff, _aff])
    size = tf.shape(aff_all)[1]
    _, indices_of_ranks = tf.nn.top_k(aff_all, k=size)
    _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
    mrr = tf.reduce_mean(tf.div(1.0, tf.cast(ranks[:, -1] + 1, tf.float32)))
    return mrr

if __name__ == "__main__":
    outputs1 = [[-0.7191, -0.6851, -0.0119, -0.1156], [-0.7191, -0.6851, -0.0119, -0.1156]]
    outputs2 = [[-0.1947, -0.1648, -0.2668, -0.9294], [-0.1111, -0.0801,  0.6420,  0.7544]]
    neg_outputs = [[-0.6569, -0.6830, -0.0261,  0.3183],
        [-0.6707, -0.4471,  0.5918,  0.0057],
        [-0.6047, -0.5160,  0.0458,  0.6050],
        [-0.8900, -0.4201,  0.1651,  0.0645],
        [-0.4245, -0.4414, -0.1310,  0.7796],
        [-0.2679, -0.2781,  0.5475, -0.7424],
        [-0.6994, -0.6184,  0.3367, -0.1229],
        [-0.1913, -0.1418,  0.0885, -0.9672],
        [-0.8257, -0.4667, -0.0679, -0.3095],
        [-0.6755, -0.3809,  0.0851,  0.6256],
        [-0.5286, -0.2899,  0.5007, -0.6212],
        [-0.7793, -0.6209,  0.0777, -0.0345],
        [-0.5586, -0.4766,  0.3483, -0.5826],
        [-0.6908, -0.5507, -0.1630, -0.4393],
        [-0.3241, -0.2744, -0.0647, -0.9030],
        [ 0.0000,  0.0000,  0.1419, -0.9899],
        [-0.8909, -0.4348,  0.1256, -0.0383],
        [-0.7795, -0.5094,  0.3273, -0.1607],
        [-0.6643, -0.5623,  0.3385, -0.3576],
        [-0.7478, -0.5858,  0.1597, -0.2685]]
    
    tf.InteractiveSession()
    mrr = accuracy(tf.constant(outputs1), tf.constant(outputs2), tf.constant(neg_outputs))
    print(mrr.eval())

    # aff = [ 0.3636,  0.0399]
    # neg_sample_size = 20
    # neg_aff = [[ 0.9038,  0.7809,  0.7178,  0.9184,  0.5191,  0.4625,  0.9368, 
    #         0.3455,  0.9501,  0.6734,  0.6446,  0.9888,  0.7915,  0.9268,
    #         0.5262,  0.1128,  0.9415,  0.9242,  0.9003,  0.9682],
    #         [ 0.9038,  0.7809,  0.7178,  0.9184,  0.5191,  0.4625,  0.9368,
    #         0.3455,  0.9501,  0.6734,  0.6446,  0.9888,  0.7915,  0.9268,
    #         0.5262,  0.1128,  0.9415,  0.9242,  0.9003,  0.9682]]
    # indices_of_rank = [[ 11,  19,   8,  16,   6,  13,  17,   3,   0,  18,  12,   1,
    #         2,   9,  10,  14,   4,   5,  20,   7,  15],
    #         [ 11,  19,   8,  16,   6,  13,  17,   3,   0,  18,  12,   1,
    #         2,   9,  10,  14,   4,   5,   7,  15,  20]]
    # ranks = [19, 21]
    # mrr = 0.05012531578540802


    
