"""Convolutional Autoencoder for CIFAR10 dataset
Qi WEI,
@Duke University
June 26th 2017
"""
import tensorflow as tf
import numpy as np
import math
from libs.activations import lrelu
from libs.utils import corrupt
from scipy.misc import imsave
import scipy.misc
from glob import glob
import os

nr = 128
nc = 128

# data_names = glob(os.path.join("/home/qw62/PSGAN/data", "CUBMR", "*.jpg"))
# data_names = glob(os.path.join("/Users/qw62/PycharmProjects/PSGAN/data", "CUBLR", "*.jpg"))
# nch = 3

# negative calcs
# data_names = glob(os.path.join("/store/usr/bs232/nobackup/calc_patch_int16_Qi", "c1", "*.png"))

# positive calcs
data_names = glob(os.path.join("/store/usr/bs232/nobackup/calc_patch_int16_Qi", "c1", "*.png"))
# data_names = glob(os.path.join("/Users/qw62/PycharmProjects/Qi_AutoEncoder/tensorflow_tutorials/python", "c0", "*.png"))


nch = 1

indices = np.arange(len(data_names))
partition = int(len(data_names) * 0.9) + 1
data_names = np.array(data_names)
train_names = data_names[indices[:partition]]
test_names = data_names[indices[partition:]]


def iterate_minibatches_u(data, batchsize, shuffle=False):
    """
    This function tries to iterate unlabeled data in mini-batch
    for batch_data in iterate_minibatches_u(data, batchsize, True):
        #processing batch_data
    """
    if shuffle:
        indices = np.arange(len(data))
        np.random.RandomState(np.random.randint(1,2147462579)).shuffle(indices)
    for start_idx in xrange(0, len(data) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield data[excerpt]


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], size[2]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img


def save_model(saver, sess, model_path, step):
    """
    save model with path error checking
    """
    if model_path is None:
        my_path = "model/myckpt"  # default path in tensorflow saveV2 format
        #  try to make directory
        if not os.path.exists("model"):
            try:
                os.makedirs("model")
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
    else:
        my_path = model_path

    saver.save(sess, my_path, global_step=step)


def autoencoder(input_shape=[None, nr, nc, nch],
                n_filters=[1, 64, 128, 256],  #n_filters=[1, 32, 64, 128],
                filter_sizes=[5, 5, 5, 5],
                corruption=False):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training

    Raises
    ------
    ValueError
        Description
    """
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')

    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # Optionally apply denoising autoencoder
    if corruption:
        current_input = corrupt(current_input)

    # Build the encoder
    encoder = []
    shapes = []
    enc_input = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        enc_input.append(current_input)
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(tf.add(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()
    enc_input.reverse()

    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        # W = encoder[layer_i]
        # defining
        n_input = current_input.get_shape().as_list()[3] #current_input.shape[3].value
        W_dec = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                shape[3], n_input],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))

        # b = tf.Variable(tf.zeros([W_dec.get_shape().as_list()[2]]))
        b = tf.Variable(tf.zeros([shape[3]]))
        output = lrelu(tf.add(tf.nn.conv2d_transpose(current_input, W_dec,
                                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]), # bs*nr*nc*nch
                                                     strides=[1, 2, 2, 1], padding='SAME'), b))
        if layer_i > -1:
            current_input = tf.concat([output, enc_input[layer_i]], 3)
            # n_input = current_input.get_shape().as_list()[3]
            n_input = shape[3]*2
            W_dec = tf.Variable(
                tf.random_uniform([
                    filter_sizes[layer_i],
                    filter_sizes[layer_i],
                    shape[3], n_input],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([shape[3]]))
            output = lrelu(tf.add(tf.nn.conv2d_transpose(current_input, W_dec,
                                                         tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]), # bs*nr*nc*nch
                                                         strides=[1, 1, 1, 1], padding='SAME'), b))
        current_input = output

    # now have the reconstruction through the network
    y = tf.sigmoid(current_input)
    # cost function measures pixel-wise difference
    cost = tf.reduce_mean(tf.square(y - x_tensor))
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


# def get_showimages(self, sess, n = 8):
#     num_show = min(n*n, self.batch_size)
#     xg = sess.run(self.x_g) # batch_size x H x W x C
#     xshow_ = np.array(xg)[:num_show,:,:,:] # num_show x H x W x C
#     xshow_ = ps_inv(xshow_, scale=4)
#     return 0.5*(xshow_+1.0)

def test_CUB():
    ae = autoencoder()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    model_path = "model/AE_CUB"
    saver.restore(sess, model_path + "-" + str(9))

    # Plot example reconstructions
    n_examples = 100
    sel_indices = np.arange(len(data_names))
    for ind in range(10):
        np.random.shuffle(sel_indices)
        index = sel_indices[:n_examples]

        hr_test = [scipy.misc.imread(img_name) for img_name in data_names[index]]
        test_xs = np.array(hr_test).astype(np.float32)  # / 2048.0
        test_org = np.reshape(test_xs, (n_examples, nr, nc, nch))
        tem_max = test_xs.max(axis=2, keepdims=True).max(axis=1, keepdims=True)
        tem_min = test_xs.min(axis=2, keepdims=True).min(axis=1, keepdims=True)
        test_xs = (test_xs - tem_min) / (tem_max - tem_min)
        test_xs = np.reshape(test_xs, (n_examples, nr, nc, nch))
        [cost_rec, recon] = sess.run([ae['cost'], ae['y']], feed_dict={ae['x']: test_xs})
        print(cost_rec)
        print([test_xs.max(), test_xs.min(), recon.max(), recon.min()])
        error = test_xs[:n_examples] - recon[:n_examples]
        imsave("results/" + str(ind) + "_input_o.png", np.squeeze(merge(test_org[:n_examples], [10, 10, nch])))
        imsave("results/" + str(ind) + "_input_p.png", np.squeeze(merge(test_xs[:n_examples], [10, 10, nch])))
        imsave("results/" + str(ind) + "_recon.png", np.squeeze(merge(recon[:n_examples], [10, 10, nch])))
        imsave("results/" + str(ind) + "_resid_org.png", np.squeeze(merge(error, [10, 10, nch])))
        tem_max = error.max(axis=2, keepdims=True).max(axis=1, keepdims=True)
        tem_min = error.min(axis=2, keepdims=True).min(axis=1, keepdims=True)
        error = (error - tem_min) / (tem_max - tem_min)
        error = np.reshape(error, (n_examples, nr, nc, nch))
        imsave("results/" + str(ind) + "_resid_scl.png", np.squeeze(merge(error, [10, 10, nch])))
    sess.close()



def train_CUB():
    """Test the convolutional autoencder using MNIST."""
    ae = autoencoder()
    saver = tf.train.Saver()
    model_path = "model/AE_CUB"

    learning_rate = 0.0005 #0.005 for CUB
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    batch_size = 200
    n_epochs = 100
    for epoch_i in range(n_epochs):
        for batch_names in iterate_minibatches_u(train_names, batch_size, shuffle=True):
            hrtrain = [scipy.misc.imread(img_name) for img_name in batch_names]
            batch_xs = np.array(hrtrain).astype(np.float32) #/ 2048.0  # batch_size x rH x rW x C
            # batch_xs = (batch_xs - batch_xs.min()) / (batch_xs.max() - batch_xs.min())
            # tem_max = batch_xs.max(axis=3, keepdims=True).max(axis=2, keepdims=True).max(axis=1, keepdims=True)
            # tem_min = batch_xs.min(axis=3, keepdims=True).min(axis=2, keepdims=True).min(axis=1, keepdims=True)
            tem_max = batch_xs.max(axis=2, keepdims=True).max(axis=1, keepdims=True)
            tem_min = batch_xs.min(axis=2, keepdims=True).min(axis=1, keepdims=True)
            batch_xs = (batch_xs - tem_min)/(tem_max - tem_min)
            train = np.reshape(batch_xs, (batch_size, nr, nc, nch))
            # train = batch_xs
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))
        save_model(saver, sess, model_path, step=epoch_i)

    # Plot example reconstructions
    n_examples = 100
    sel_indices = np.arange(len(test_names))
    np.random.shuffle(sel_indices)
    index = sel_indices[:n_examples]

    hr_test = [scipy.misc.imread(img_name) for img_name in test_names[index]]
    test_xs = np.array(hr_test).astype(np.float32) #/ 2048.0
    # test_xs = (test_xs - test_xs.min()) / (test_xs.max() - test_xs.min())
    tem_max = test_xs.max(axis=2, keepdims=True).max(axis=1, keepdims=True)
    tem_min = test_xs.min(axis=2, keepdims=True).min(axis=1, keepdims=True)
    test_xs = (test_xs - tem_min) / (tem_max - tem_min)
    test_xs = np.reshape(test_xs, (n_examples, nr, nc, nch))
    [cost_rec, recon] = sess.run([ae['cost'], ae['y']], feed_dict={ae['x']: test_xs})
    print(cost_rec)
    print(recon.shape)
    print([test_xs.max(), test_xs.min(), recon.max(), recon.min()])
    imsave("results/test.png", np.squeeze(merge(test_xs[:n_examples], [10, 10, nch])))
    imsave("results/recon.png", np.squeeze(merge(recon[:n_examples], [10, 10, nch])))
    imsave("results/error.png", np.squeeze(merge(test_xs[:n_examples]-recon[:n_examples], [10, 10, nch])))

if __name__ == '__main__':
    train_CUB()
    # test_CUB()