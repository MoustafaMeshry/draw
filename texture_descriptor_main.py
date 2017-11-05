from skimage.color import rgb2gray
from texture_loss import filter_response2histogram
from texture_loss import im2filter_response
import glob
import numpy as np
import os.path as osp
import pickle as pkl
import scipy.io as sio
import skimage.io
import tensorflow as tf


def toy_example():
    # Load images and convert to grayscale
    img1 = skimage.io.imread('texture/trivial/0.jpg')
    # img2 = skimage.io.imread('texture/trivial/1.jpg')
    img2 = skimage.io.imread('texture/trivial/0.jpg')
    img1 = rgb2gray(img1)
    img2 = rgb2gray(img2)
    img1 = img1[0:50, 0:50]
    img2 = img2[0:50, 0:50]

    # Construct mini-batch
    img1 = img1.reshape(img1.shape + (1,)).astype(np.float32)
    img2 = img2.reshape(img2.shape + (1,)).astype(np.float32)
    mini_batch = np.stack((img1, img2))

    # Load filters
    filter_kernel = pkl.load(open('filters/np_LM_filter_p2.pkl', 'rb'))
    lm_centroids = pkl.load(open('filters/np_centroids_p2.pkl', 'rb'))
    lm_centroids = lm_centroids.astype(np.float32)

    # Change filters to 4D (for convolution)
    filter_kernel = filter_kernel.reshape((49, 49, 1, 48)).astype(np.float32)

    # Build operations
    filter_response_op = im2filter_response(
        tf.convert_to_tensor(mini_batch), tf.convert_to_tensor(filter_kernel))

    # centroids_op = tf.convert_to_tensor(lm_centroids)
    num_bins = 20
    hists_op = filter_response2histogram(filter_response_op, lm_centroids,
                                         num_bins, 2)

    # Run im2filter_response in tensorflow
    with tf.Session() as sess:
        filter_response, hists = sess.run([filter_response_op, hists_op])
    print(hists)


def debug_descriptor(input_path, file_ext, num_bins):
    path = osp.join(input_path, '*.%s' % file_ext)
    output_path = 'data/debug_histograms.mat'

    hists_all = []
    sess = tf.InteractiveSession()
    for f in glob.glob(path):
        img = skimage.io.imread(f)
        img = rgb2gray(img)
        # img = img[0:50, 0:50]
        assert img.shape[0] < 100 and img.shape[1] < 100, (
            "Image size is too big and might cause memory problems in KNN")

        # Reshape to 1xHxWx1 (a mini_batch of size 1)
        mini_batch = img.reshape((1,) + img.shape + (1,)).astype(np.float32)
        mini_batch = tf.convert_to_tensor(mini_batch)

        # Load filters
        filter_kernel = pkl.load(open('filters/np_LM_filter_p2.pkl', 'rb'))
        lm_centroids = pkl.load(open('filters/np_centroids_p2.pkl', 'rb'))
        lm_centroids = lm_centroids.astype(np.float32)

        # Change filters to 4D (for convolution)
        filter_kernel = filter_kernel.reshape((49, 49, 1, 48)).astype(
            np.float32)
        filter_kernel_tf = tf.convert_to_tensor(filter_kernel)

        # Build operations
        filter_response_op = im2filter_response(
                            mini_batch, tf.convert_to_tensor(filter_kernel_tf))

        hists_op = filter_response2histogram(filter_response_op, lm_centroids,
                                             num_bins, 1)

        # Run im2filter_response in tensorflow
        hist = sess.run(hists_op)
        hists_all.append(np.squeeze(hist))

    hist_dict = {'histograms': hists_all}
    sio.savemat(output_path, hist_dict)

    sess.close()


def main(_):
    # toy_example()
    dataset_path = 'texture/trivial'
    file_ext = 'jpg'
    num_bins = 200
    debug_descriptor(dataset_path, file_ext, num_bins)


if __name__ == '__main__':
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.app.run()
