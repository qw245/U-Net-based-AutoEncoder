"""Convolutional Autoencoder
Qi WEI, 2017
"""
import tensorflow as tf
import numpy as np
import math
from libs.activations import lrelu
from libs.utils import corrupt
from scipy.misc import imsave

nr = 28
nc = 28
nch = 1
n_samples = 50000


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], size[2]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img


def autoencoder(input_shape=[None, nr*nc*nch],
                n_filters=[1, 16, 32, 64],
                filter_sizes=[3, 3, 3, 3],
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
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')

    # %%
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
    for layer_i, n_output in enumerate(n_filters[1:]):
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

    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(tf.nn.conv2d_transpose(current_input, W,
                                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                                     strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_mean(tf.square(y - x_tensor))
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


def test_mnist():
    """Test the convolutional autoencder using MNIST."""
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # mean_img = np.mean(mnist.train.images, axis=0)*0
    ae = autoencoder()

    learning_rate = 0.005
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    batch_size = 100
    n_epochs = 50
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            # train = np.array([img - mean_img for img in batch_xs])
            train = np.array(batch_xs)
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # Plot example reconstructions
    n_examples = 100
    test_xs, _ = mnist.test.next_batch(n_examples)
    # test_xs_norm = np.array([img - mean_img for img in test_xs])
    test_xs_norm = np.array(test_xs)
    [cost_rec, recon] = sess.run([ae['cost'], ae['y']], feed_dict={ae['x']: test_xs_norm})
    print(cost_rec)
    print(recon.shape)


    # plot part
    # fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    test_xs = np.reshape(test_xs, (n_examples, nr, nc, nch))
    recon = np.reshape(recon, (n_examples, nr, nc, nch))
    imsave("results/test.jpg", np.squeeze(merge(test_xs[:n_examples], [10, 10, nch])))
    imsave("results/recon.jpg", np.squeeze(merge(recon[:n_examples], [10, 10, nch])))
    imsave("results/error.jpg", np.squeeze(merge(test_xs[:n_examples]-recon[:n_examples], [10, 10, nch])))
    # for example_i in range(n_examples):
    #     imsave("results/test/test_"+str(example_i)+'.jpg', np.reshape(test_xs[example_i, :], (28, 28)))
    #     imsave("results/recon/recon_" + str(example_i) + '.jpg', np.reshape(
    #             np.reshape(recon[example_i, ...], (784,)) + mean_img,
    #             (28, 28)))
    #     # axs[0][example_i].imshow(
    #     #     np.reshape(test_xs[example_i, :], (28, 28)))
    #     # axs[1][example_i].imshow(
    #     #     np.reshape(
    #     #         np.reshape(recon[example_i, ...], (784,)) + mean_img,
    #     #         (28, 28)))
    # plt.close()
    # fig.show()
    # plt.draw()
    # plt.waitforbuttonpress()

if __name__ == '__main__':
    test_mnist()
