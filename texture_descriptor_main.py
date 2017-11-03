from skimage.color import rgb2gray
import numpy as np
import pickle as pkl
import skimage.io
import tensorflow as tf


def im2filter_response(imgs, filter_kernel_4d, hps=None):
    """
    constructs texture filter response for a mini-batch of grayscale images
    imgs: NxHxWx1 batch of *GRAYSCALE* images
    filter_kernel: 4-D filter kernel [height x width x 1 x num_output_channels]
    hps: Hyper params. I only need the batch size, image patch heigt and width
    returns NxHxWxC tensor of C-channels of filter responses
    """
    # num_channels = filter_kernel_4d.get_shape()[-1]
    mini_batch_shape = imgs.get_shape()
    [n_batch, height, width, _] = mini_batch_shape

    response = tf.nn.conv2d(imgs, filter_kernel_4d, strides=[1, 1, 1, 1],
                            padding='SAME')
    # response_norm = tf.norm(response, axis=3, keep_dims=False)  # o/p NxHxW
    response_norm = tf.norm(response, axis=3, keep_dims=True)  # o/p NxHxWx1
    sc = tf.log(1 + (response_norm / 0.03))

# No need for tiling. Array broadcasting will do the job
    # sc_tile = tf.tile(sc, [1, num_channels, 1])
    # sc_tile = tf.reshape(sc_tile, [n_batch, height, width, num_channels])
    # sc_tile = tf.transpose(sc_tile, [0, 2, 3, 1])
    #
    # response_norm_tile = tf.tile(response_norm, [1, num_channels, 1])
    # response_norm_tile = tf.reshape(response_norm_tile,
    #                                 [n_batch, height, width, num_channels])
    # response_norm_tile = tf.transpose(response_norm_tile, [0, 2, 3, 1])

    # numerator = tf.multiply(response, sc_tile)
    # return tf.divide(numerator, response_norm_tile)

    numerator = tf.multiply(response, sc)
    return tf.divide(numerator, response_norm)


# Load images and convert to grayscale
img1 = skimage.io.imread('texture/trivial/0.jpg')
# img2 = skimage.io.imread('texture/trivial/1.jpg')
img2 = skimage.io.imread('texture/trivial/0.jpg')  # i need same-sized images
img1 = rgb2gray(img1)
img2 = rgb2gray(img2)

# Construct mini-batch
img1 = img1.reshape(img1.shape + (1,)).astype(np.float32)
img2 = img2.reshape(img2.shape + (1,)).astype(np.float32)
mini_batch = np.stack((img1, img2))

# Load filters
filter_kernel = pkl.load(open('filters/np_LM_filter_p2.pkl', 'rb'))

# Change filters to 4D (for convolution)
filter_kernel = filter_kernel.reshape((49, 49, 1, 48)).astype(np.float32)

# Build operations
filter_response_op = im2filter_response(tf.convert_to_tensor(mini_batch),
                                        tf.convert_to_tensor(filter_kernel))

# Run im2filter_response in tensorflow
with tf.Session() as sess:
    filter_response = sess.run(filter_response_op)
