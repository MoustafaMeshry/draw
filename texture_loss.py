import constants as const
import numpy as np
import pickle as pkl
import tensorflow as tf


class TextureLoss:

    def __init__(self, filter_file_path, centroids_file_path, num_bins,
                 batch_sz):
        """
        filter_file_path: path to pickled LM filter
        centroids_file_path: path to pickled filter response centroids
        num_bins: number of bins for the output texture histogram
        batch_sz: mini_batch size
        """
        self.num_bins = num_bins
        self.batch_sz = batch_sz

        # Load filters
        filter_kernel = pkl.load(open(filter_file_path, 'rb'))
        centroids = pkl.load(open(centroids_file_path, 'rb'))
        centroids = centroids.astype(np.float32)

        # Change LM filter to 4D (for convolution)
        filter_kernel = filter_kernel.reshape((49, 49, 1, 48)).astype(
            np.float32)
        self.filter_tf = tf.convert_to_tensor(filter_kernel)
        self.centroids_numpy = centroids

    def binary_crossentropy(self, t, o):
        return -(t * tf.log(o + const.eps) + (
                 1.0 - t) * tf.log(1.0 - o + const.eps))

    def texture_filter_bank_loss(self, y, y_gt):
        y_filter_response = im2filter_response(y, self.filter_tf)
        y_gt_filter_response = im2filter_response(y_gt, self.filter_tf)

        y_hist = filter_response2histogram(
            y_filter_response, self.centroids_numpy, self.num_bins,
            self.batch_sz)
        y_gt_hist = filter_response2histogram(
            y_gt_filter_response, self.centroids_numpy, self.num_bins,
            self.batch_sz)

        # l2_loss = tf.reduce_mean(tf.nn.l2_loss(y_hist - y_gt_hist))
        l2_loss = tf.nn.l2_loss(y_hist - y_gt_hist)
        return l2_loss


def im2filter_response(imgs, filter_kernel_4d):
    """
    constructs texture filter response for a mini-batch of grayscale images
    imgs: NxHxWx1 batch of *GRAYSCALE* images
    filter_kernel: 4-D filter kernel [height x width x 1 x num_channels]
    returns NxHxWxC tensor of C-channels of filter responses
    """
    # num_channels = filter_kernel_4d.get_shape()[-1]
    mini_batch_shape = imgs.get_shape()
    [n_batch, height, width, _] = mini_batch_shape

    response = tf.nn.conv2d(imgs, filter_kernel_4d, strides=[1, 1, 1, 1],
                            padding='SAME')
    response_norm = tf.norm(response, axis=3, keep_dims=True)  # NxHxWx1
    sc = tf.log(1 + (response_norm / 0.03))

    numerator = tf.multiply(response, sc)
    return tf.divide(numerator, response_norm)


def filter_response2histogram(filter_responses, training_class_centroids,
                              num_bins, batch_sz):
    """
    Builds texture descriptor (normalized histogram) from filter response
    filter_responses: NxHxWxC tensor where C is number of filter channels
    training_class_centroids: NxC NumPy array. N is # of centroids
    num_bins: number of bins in the output histogram (MUST be a divisor of
    the number of centroids)
    batch_sz: mini_batch size
    """

    # 1) Compute KNN for each pixel in each image in the mini_batch

    num_centroids, response_sz = training_class_centroids.shape
    training_class_centroids = tf.convert_to_tensor(training_class_centroids)

    queries = tf.reshape(filter_responses, [batch_sz, -1, 1, response_sz])
    centroids = tf.reshape(training_class_centroids, [1, 1, num_centroids, -1])

    diff = tf.square(queries - centroids)
    dist = tf.reduce_sum(diff, axis=3)
    # knn = tf.argmin(dist, axis=2, output_type=tf.int32)  #TF v1.4
    knn = tf.argmin(dist, axis=2)
    knn = tf.cast(knn, tf.float32)

    # 2) Compute a histogram for each image. NxNUM_BINS

    # The following is a work around to compute separate histograms for
    # each image in the mini_batch
    batch_shifter = tf.range(0, batch_sz, dtype=tf.float32) * num_centroids
    batch_shifter = tf.expand_dims(batch_shifter, -1)  # for broadcasting
    knn = knn + batch_shifter
    histograms_flattened = tf.histogram_fixed_width(
        knn, [0.0, num_centroids * batch_sz * 1.0], num_bins * batch_sz)
    histograms = tf.reshape(histograms_flattened, [batch_sz, -1])
    histograms = tf.cast(histograms, tf.float32)

    # 3) Normalize each histogram by dividing it by its sum

    hist_sums = tf.reduce_sum(histograms, axis=1, keep_dims=True)
    normalized_histograms = histograms / hist_sums

    return normalized_histograms
