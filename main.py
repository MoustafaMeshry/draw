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
    size = const.A
    with_attention= False
    save_path = os.path.join("./train/",'tmp_simple_d'+str(direction)+'_s'+str(size)+'_a'+str(with_attention));
    tf.flags.DEFINE_string("data_dir", save_path , "")
    tf.flags.DEFINE_boolean("read_attn", with_attention, "enable attention for reader")
    tf.flags.DEFINE_boolean("write_attn", with_attention, "enable attention for writer")
    FLAGS = tf.flags.FLAGS


    model = draw_model.DrawModel(with_attention,with_attention);
    mus = model.mus
    logsigmas = model.logsigmas
    sigmas = model.sigmas

    batch_sz = const.batch_size
    num_bins = 200;
    filter_file_path = './filters/np_LM_filter_p2.pkl'
    centroids_file_path = './filters/np_centroids_p2.pkl'

    filter_bank_loss = texture_loss.TextureLoss(filter_file_path, centroids_file_path, num_bins,batch_sz);

    # reconstruction term appears to have been collapsed down to a single scalar
    # value (rather than one per item in minibatch)
    y_recons = tf.nn.sigmoid(model.cs[-1])

    # after computing binary cross entropy, sum across features then take the
    # mean of those sums across minibatches

    #Lx = tf.reduce_sum(filter_bank_loss.binary_crossentropy(model.y, y_recons ), 1)  # reconstruction term
    #Lx = tf.reduce_mean(Lx)

    #Lx = filter_bank_loss.gaussian_loss(model.y, y_recons)  # reconstruction term

    Lx = filter_bank_loss.mean_color_loss(model.y, y_recons)  # reconstruction term

    Lz = model.Lz


    Lc = filter_bank_loss.texture_filter_bank_loss(model.y, y_recons)  # reconstruction term
    #hist = filter_bank_loss.hist;
    #Lc = Lc;

    #Lz = tf.divide(model.Lz,10)
    # Cost conatins two parts
    ## 1. Cost from latent variable distribution
    ## 2. Cost from latent variable distribution


    cost1 = Lx
    cost2 = Lz + Lc
    #print(tf.shape(Lc))
    #print(tf.shape(Lz))
    #print(tf.shape(Lx))


    # ==OPTIMIZER== #
    learning_rate = 1e-3
    print('learning_rate ',learning_rate )
    optimizer1 = tf.train.AdamOptimizer(learning_rate , beta1=0.5)
    optimizer2 = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

    grads1 = optimizer1.compute_gradients(cost1)
    grads2 = optimizer2.compute_gradients(cost2)

    for i, (g, v) in enumerate(grads1):
        if g is not None:
            grads1[i] = (tf.clip_by_norm(g, 5), v)
    train_op1 = optimizer1.apply_gradients(grads1)

    for i, (g, v) in enumerate(grads2 ):
        if g is not None:
            grads2[i] = (tf.clip_by_norm(g, 5), v)
    train_op2 = optimizer2.apply_gradients(grads2)
    # ==RUN TRAINING== #

    fetches1 = []
    fetches1.extend([Lx, Lz,Lc, train_op1])
    fetches2 = []
    fetches2.extend([Lx, Lz, Lc, train_op2])
    #fetches.extend([Lx, Lz, train_op])
    Lxs = [0] * const.train_iters
    Lzs = [0] * const.train_iters

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()  # saves variables learned during training
    tf.global_variables_initializer().run()

    # to restore from model, uncomment the next line
    ckpt_file = os.path.join(FLAGS.data_dir, "drawmodel.ckpt")
    print(ckpt_file)
    if(os.path.exists(FLAGS.data_dir) and len(os.listdir(FLAGS.data_dir))> 0):
        saver.restore(sess, ckpt_file )
        print('Previous Model loaded ',len(os.listdir(FLAGS.data_dir)))
    else:
        print('Learning from scratch')
    img_generator = batch_gen.BatchGenerator(const.batch_size, FLAGS.data_dir)

    print(FLAGS.data_dir, os.path.exists(FLAGS.data_dir))
    if (os.path.exists(FLAGS.data_dir) == False):
        os.makedirs(FLAGS.data_dir)

    for i in range(const.train_iters):
        # xtrain is (batch_size x img_size)
        xtrain, ytrain = img_generator.next(direction=direction)
        feed_dict = {model.x: xtrain, model.y: ytrain}

        results = sess.run(fetches1, feed_dict)
        Lxs[i], Lzs[i], lc ,_ = results
        print("Fet1 iter=%d : Lx: %f Lz: %f  Lc %f" % (i, Lxs[i], Lzs[i], lc))

        results = sess.run(fetches2, feed_dict)
        Lxs[i], Lzs[i], lc, _ = results
        print("Fet2 iter=%d : Lx: %f Lz: %f  Lc %f" % (i, Lxs[i], Lzs[i], lc))

        #Lxs[i], Lzs[i],_ = results
        #print("iter=%d : Lx: %f Lz: %f " % (i, Lxs[i], Lzs[i]))
        if i != 0 and i % 100 == 0:

            #if (not const.gpu_used):
            m,s,logs = sess.run([mus ,sigmas,logsigmas],feed_dict)
            print('mean ',np.mean(m[-1]))
            print('std ', np.mean(s[-1]))
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