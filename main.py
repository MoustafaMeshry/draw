import draw_model
import batch_generator as batch_gen
import numpy as np
import os
import tensorflow as tf
import texture_loss
import constants as const

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.flags.DEFINE_string("data_dir", "./train/texture_simple_model/", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn", True, "enable attention for writer")
FLAGS = tf.flags.FLAGS



if __name__ == '__main__':
    model = draw_model.DrawModel(FLAGS);
    filter_bank_loss = texture_loss.TextureLoss();

    # reconstruction term appears to have been collapsed down to a single scalar
    # value (rather than one per item in minibatch)
    y_recons = tf.nn.sigmoid(model.cs[-1])

    # after computing binary cross entropy, sum across features then take the
    # mean of those sums across minibatches
    Lx = tf.reduce_sum(filter_bank_loss.texture_filter_bank_loss(model.y, y_recons ), 1)  # reconstruction term
    Lx = tf.reduce_mean(Lx)

    # Cost conatins two parts
    ## 1. Cost from latent variable distribution
    ## 2. Cost from latent variable distribution
    Lz = model.Lz
    cost = Lx + Lz


    # ==OPTIMIZER== #

    optimizer = tf.train.AdamOptimizer(const.learning_rate, beta1=0.5)
    grads = optimizer.compute_gradients(cost)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
    train_op = optimizer.apply_gradients(grads)

    # ==RUN TRAINING== #

    fetches = []
    fetches.extend([Lx, Lz, train_op])
    Lxs = [0] * const.train_iters
    Lzs = [0] * const.train_iters

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()  # saves variables learned during training
    tf.global_variables_initializer().run()

    # to restore from model, uncomment the next line
    # saver.restore(sess, "/tmp/draw/drawmodel.ckpt")
    img_generator = batch_gen.BatchGenerator(const.batch_size, FLAGS.data_dir)

    for i in range(const.train_iters):
        # xtrain is (batch_size x img_size)
        xtrain, ytrain = img_generator.next()
        feed_dict = {model.x: xtrain, model.y: ytrain}
        results = sess.run(fetches, feed_dict)
        Lxs[i], Lzs[i], _ = results
        if i % 100 == 0:
            print("iter=%d : Lx: %f Lz: %f" % (i, Lxs[i], Lzs[i]))

    # ==TRAINING FINISHED== #

    canvases = sess.run(model.cs, feed_dict)  # generate some examples
    canvases = np.array(canvases)  # T x batch x img_size

    print(FLAGS.data_dir, os.path.exists(FLAGS.data_dir))
    if (os.path.exists(FLAGS.data_dir) == False):
        os.makedirs(FLAGS.data_dir)

    out_file = os.path.join(FLAGS.data_dir, "draw_data.npy")
    print(canvases.shape, len(Lxs), len(Lzs))
    np.save(out_file, [canvases, Lxs, Lzs])
    print("Outputs saved in file: %s" % out_file)

    ckpt_file = os.path.join(FLAGS.data_dir, "drawmodel.ckpt")
    print("Model saved in file: %s" % saver.save(sess, ckpt_file))

    sess.close()