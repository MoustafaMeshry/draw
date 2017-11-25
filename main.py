import draw_model
import batch_generator as batch_gen
import numpy as np
import os
import tensorflow as tf
import texture_loss
import constants as const


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    direction = const.Direction.LEFT.value
    with_attention= const.attention_flag
    save_path = os.path.join("./train/", 'simple_d' + str(direction) + '_s' + \
        str(const.A) + '_a' + str(with_attention))
    tf.flags.DEFINE_string("data_dir", save_path , "")
    tf.flags.DEFINE_boolean("read_attn", with_attention,
                            "enable attention for reader")
    tf.flags.DEFINE_boolean("write_attn", with_attention,
                            "enable attention for writer")
    FLAGS = tf.flags.FLAGS

    batch_sz = const.batch_size
    num_bins = 200;
    filter_file_path = './filters/np_LM_filter_p2.pkl'
    centroids_file_path = './filters/np_centroids_p2.pkl'

    model = draw_model.DrawModel(FLAGS.read_attn, FLAGS.write_attn);
    filter_bank_loss = texture_loss.TextureLoss(
                    filter_file_path, centroids_file_path, num_bins,batch_sz);

    # FIXME this won't play nice with RGB continuous values!
    y_recons = tf.nn.sigmoid(model.cs[-1])
    # y_recons = model.cs[-1]

    variational_loss_weight = 1  # 1.0 / (10*2*8)

    # Reconstruction loss
    # Cross entropy loss
    Lx = tf.reduce_sum(filter_bank_loss.binary_crossentropy(model.y, y_recons), 1)
    Lx = tf.reduce_mean(Lx)
    
    # L2 loss
    # Lx = filter_bank_loss.l2_loss(model.y, y_recons)

    # # Filter-bank loss
    # Lx = filter_bank_loss.texture_filter_bank_loss(model.y, y_recons)

    # # Vgg loss
    # Lx = filter_bank_loss.vgg_loss(model.y, y_recons)

    # Variational loss (for latent variable z)
    Lz = model.Lz
    # Lz = tf.divide(model.Lz, 10*2*8)

    # Cost conatins two parts:
    ## 1. Cost from image reconstruction (generation)
    ## 2. Cost from latent variable distribution
    cost = Lx + variational_loss_weight * Lz


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

    print(FLAGS.data_dir, os.path.exists(FLAGS.data_dir))
    if (os.path.exists(FLAGS.data_dir) == False):
        os.makedirs(FLAGS.data_dir)

    for i in range(const.train_iters):
        # xtrain is (batch_size x img_size)
        xtrain, ytrain = img_generator.next(direction=direction)
        feed_dict = {model.x: xtrain, model.y: ytrain}
        results = sess.run(fetches, feed_dict)
        Lxs[i], Lzs[i], _ = results
        if i != 0 and i % 100 == 0:
            print("iter=%d : Lx: %f Lz: %f" % (i, Lxs[i], Lzs[i]))
            if (i % 1000 == 0):
                ckpt_file = os.path.join(FLAGS.data_dir, "drawmodel.ckpt")
                print("Model saved in file: %s" % saver.save(sess, ckpt_file))

                out_file = os.path.join(FLAGS.data_dir, "draw_data.npy")
                np.save(out_file, [Lxs[:i], Lzs[:i]])
                print("Outputs saved in file: %s" % out_file)

    # ==TRAINING FINISHED== #

    out_file = os.path.join(FLAGS.data_dir, "draw_data.npy")
    np.save(out_file, [Lxs, Lzs])
    print("Outputs saved in file: %s" % out_file)


    ckpt_file = os.path.join(FLAGS.data_dir, "drawmodel.ckpt")
    print("Model saved in file: %s" % saver.save(sess, ckpt_file))

    sess.close()
