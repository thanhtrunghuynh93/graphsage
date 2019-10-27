import tensorflow as tf

def affinity(inputs1, inputs2):
    result = tf.reduce_sum(inputs1 * inputs2, axis=1)
    return result

def neg_cost(inputs1, neg_samples, hard_neg_samples=None):
    neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))
    return neg_aff

def loss(inputs1, inputs2, neg_samples):
    aff = affinity(inputs1, inputs2)
    neg_aff = neg_cost(inputs1, neg_samples)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
    # print(true_xent.eval())
    # sig_aff = tf.nn.sigmoid(aff)
    # labels = tf.ones_like(aff)
    # lo = labels * -tf.log(sig_aff) + (1 - labels) * -tf.log(1 - sig_aff)
    # print(lo.eval())
    negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)
    loss = tf.reduce_sum(true_xent) + 1.0 * tf.reduce_sum(negative_xent)
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
        
    tf.InteractiveSession()
    l = loss(tf.constant(inputs1), tf.constant(inputs2), tf.constant(neg_samples))
    print(l.eval())
    
    # true_aff = [ 0.8923,  0.8923]
    # neg_aff = [[ 0.9406,  0.9110,  0.9206,  0.8939,  0.4920,  0.9440,  0.9428,
    #             0.9270,  0.9203,  0.9632,  0.9001,  0.9226,  0.9573,  0.7418,
    #             0.9424,  0.9454,  0.4807,  0.7854,  0.9569,  0.8820],
    #             [ 0.9406,  0.9110,  0.9206,  0.8939,  0.4920,  0.9440,  0.9428,
    #             0.9270,  0.9203,  0.9632,  0.9001,  0.9226,  0.9573,  0.7418,
    #             0.9424,  0.9454,  0.4807,  0.7854,  0.9569,  0.8820]]
    # true_xent = 0.3434
    # neg_xent = 1.2209
    # loss = 1.5643