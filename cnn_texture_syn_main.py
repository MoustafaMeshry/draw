import argparse
import batch_generator as batch_gen
import constants as const
import custom_vgg19 as vgg19
import cv2
import logging
import numpy as np
import os
import tensorflow as tf
import time
import utils
from functools import reduce
from synthesize import *


# Model hyperparams
TEXTURE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
EPOCHS = 6000
LEARNING_RATE = .02
TOTAL_VARIATION_SMOOTHING = 1.5
NORM_TERM = 6.

# Loss term weights
TEXTURE_WEIGHT = 3.
NORM_WEIGHT = .1
TV_WEIGHT = .1  # TODO: Meshry: I believe results with TV_WEIGHT=0.5 are better

# Default image paths
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# OUT_PATH = DIR_PATH + '/../output/out_%.0f.jpg' % time.time()
INPUT_PATH, TEXTURE_PATH = None, None

# Logging params
PRINT_TRAINING_STATUS = True
PRINT_N = 100

# Logging config
log_dir = DIR_PATH + '/../log/'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
    print('Directory "%s" was created for logging.' % log_dir)
log_path = ''.join([log_dir, str(time.time()), '.log'])
logging.basicConfig(filename=log_path, level=logging.INFO)
print("Printing log to %s" % log_path)


if __name__ == '__main__':
    with tf.Session() as sess:
        # parse_args()
        batch_sz = 1
        img_generator = batch_gen.BatchGenerator(batch_sz, '')

        texture_model = vgg19.Vgg19()
        x_model = vgg19.Vgg19()
        for i in range(1):
    
            image_shape = [const.B, const.A, 3]
            texture, _ = img_generator.next(direction=const.Direction.LEFT.value)
            # texture, image_shape = utils.load_image('./texture/simple/1.jpg')
            # texture, image_shape = utils.load_image('./texture/simple/%d.jpg' % i)
            image_shape = [1] + image_shape
            texture = texture.reshape(image_shape).astype(np.float32)
            print('DBG: ', texture.shape)
            # print('DBG: (max,min) = ', np.max(texture), np.min(texture))

            out_dir = os.path.dirname(os.path.realpath(__file__)) + '/../output/'
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            inp_save_path = DIR_PATH + '/../output/input_%d.jpg' % i;
            inp_img = texture.reshape(image_shape[1:]) * 255
            inp_img = inp_img.astype(np.int)
            cv2.imwrite(inp_save_path, inp_img)
            # print('DBG: inp_img shape = ', inp_img.shape, type(inp_img), type(inp_img[0][0][0]))
            print('DBG: saved input image to %s!' % inp_save_path)
            OUT_PATH = DIR_PATH + '/../output/out_%d_%.0f.jpg' % (i, time.time())

            # Initialize variable image that'll become our final output as random noise
            # TODO: try initializing noise to zeros, to see if this would still work similar to draw
            noise_init = tf.truncated_normal(image_shape, mean=.5, stddev=.1)
            # noise_init = tf.zeros(image_shape)
            noise = tf.Variable(noise_init, dtype=tf.float32)

            # with tf.name_scope('vgg_texture'):
            #     texture_model = vgg19.Vgg19()
            # texture_model.build(texture, image_shape[1:], isBGR=True)
            texture_model.build(texture, image_shape, isBGR=True)
            # texture_model.build(255 * texture, image_shape[1:])

            # with tf.name_scope('vgg_x'):
            #     x_model = vgg19.Vgg19()
            # x_model.build(noise, image_shape[1:], isBGR=False)
            x_model.build(noise, image_shape, isBGR=False)

            # Loss functions
            with tf.name_scope('loss'):
                # Texture
                if TEXTURE_WEIGHT is 0:
                    texture_loss = tf.constant(0.)
                else:
                    unweighted_texture_loss = get_texture_loss(x_model, texture_model)
                    texture_loss = unweighted_texture_loss * TEXTURE_WEIGHT

                # Norm regularization
                if NORM_WEIGHT is 0:
                    norm_loss = tf.constant(0.)
                else:
                    norm_loss = (get_l2_norm_loss(noise) ** NORM_TERM) * NORM_WEIGHT

                # Total variation denoising
                if TV_WEIGHT is 0:
                    tv_loss = tf.constant(0.)
                else:
                    tv_loss = get_total_variation(noise, image_shape) * TV_WEIGHT

                # Total loss
                total_loss = texture_loss + norm_loss + tv_loss

            # Update image
            with tf.name_scope('update_image'):
                optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                grads = optimizer.compute_gradients(total_loss, [noise])
                clipped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
                update_image = optimizer.apply_gradients(clipped_grads)

            # Train
            logging.info("Initializing variables and beginning training..")
            sess.run(tf.global_variables_initializer())
            start_time = time.time()
            for i in range(EPOCHS):
                _, loss = sess.run([update_image, total_loss])
                if PRINT_TRAINING_STATUS and i % PRINT_N == 0:
                    logging.info("Epoch %04d | Loss %.03f" % (i, loss))

            # FIN
            elapsed = time.time() - start_time
            logging.info("Training complete. The session took %.2f seconds to complete." % elapsed)
            logging.info("Rendering final image and closing TensorFlow session..")

            # Render the image after making sure the repo's dedicated output dir exists
            utils.render_img(sess, noise, save=True, out_path=OUT_PATH)

