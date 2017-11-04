import draw_model
import batch_generator as batch_gen
import numpy as np
import os
import tensorflow as tf
import constants as const
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.flags.DEFINE_string("data_dir", "./train/texture_simple_model/", "Where to save model after training")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn", True, "enable attention for writer")
FLAGS = tf.flags.FLAGS


def xrecons_grid(X, B, A):
    """
    plots canvas for single time step
    X is x_recons, (batch_size x img_size)
    assumes features = BxA images
    batch is assumed to be a square number
    """
    padsize = 1
    padval = .5
    ph = B + 2 * padsize
    pw = A + 2 * padsize
    batch_size = X.shape[0]
    N = int(np.sqrt(batch_size))
    X = X.reshape((N, N, B, A))
    img = np.ones((N * ph, N * pw)) * padval
    for i in range(N):
        for j in range(N):
            startr = i * ph + padsize
            endr = startr + B
            startc = j * pw + padsize
            endc = startc + A
            img[startr:endr, startc:endc] = X[i, j, :, :]
    return img

if __name__ == '__main__':
    # load module

    model = draw_model.DrawModel(FLAGS);



    sess = tf.InteractiveSession()
    saver = tf.train.Saver()  # saves variables learned during training
    tf.global_variables_initializer().run()

    # to restore from model, uncomment the next line
    ckpt_file = os.path.join(FLAGS.data_dir, "drawmodel.ckpt")
    saver.restore(sess, ckpt_file)


    # enter images
    img_generator = batch_gen.BatchGenerator(const.batch_size, FLAGS.data_dir)
    xtrain, ytrain = img_generator.next()
    feed_dict = {model.x: xtrain, model.y: ytrain}
    canvases = sess.run(model.cs, feed_dict)  # generate some examples
    canvases = np.array(canvases)  # T x batch x img_size

    # visualize results
    T, batch_size, img_size = canvases.shape
    X = 1.0 / (1.0 + np.exp(-canvases))  # x_recons = sigmoid(canvas)
    B = A = int(np.sqrt(img_size))
    prefix = './output/myattn_deploy'

    for t in range(T):
        img = xrecons_grid(X[t, :, :], B, A)

        plt.matshow(img, cmap=plt.cm.gray)
        # you can merge using imagemagick, i.e. convert -delay
        # 10 -loop 0 *.png mnist.gif
        imgname = '%s_%d.png' % (prefix, t)
        plt.savefig(imgname)
        print(imgname)

    f = plt.figure()
    #plt.plot(Lxs, label='Reconstruction Loss Lx')
    #plt.plot(Lzs, label='Latent Loss Lz')
    plt.xlabel('iterations')
    plt.legend()