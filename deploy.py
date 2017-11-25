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
import cv2
from scipy import ndimage


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    # load module

    direction = const.Direction.DOWN.value
    size = const.A
    with_attention = const.attention_flag
    save_path = os.path.join("./train/", 'simple_' + str(direction) + '_s' + str(size) + '_a' + str(with_attention));
    tf.flags.DEFINE_string("data_dir", save_path, "")
    tf.flags.DEFINE_boolean("read_attn", with_attention, "enable attention for reader")
    tf.flags.DEFINE_boolean("write_attn", with_attention, "enable attention for writer")
    FLAGS = tf.flags.FLAGS

    is_result_sharpen = False
    model = draw_model.DrawModel(FLAGS.read_attn, FLAGS.write_attn);



    sess = tf.InteractiveSession()
    saver = tf.train.Saver()  # saves variables learned during training
    tf.global_variables_initializer().run()

    # to restore from model, uncomment the next line
    ckpt_file = os.path.join(FLAGS.data_dir, "drawmodel.ckpt")
    saver.restore(sess, ckpt_file)


    # enter images
    img_generator = batch_gen.BatchGenerator(const.batch_size, FLAGS.data_dir)

    xtrain, ytrain = img_generator.next(direction=direction,debug = True)

    print('xtrain.shape ',xtrain.shape)
    #sys.exit(1)
    feed_dict = {model.x: xtrain, model.y: ytrain}
    canvases = sess.run(model.cs, feed_dict)  # generate some examples
    canvases = np.array(canvases)  # T x batch x img_size

    # visualize results
    T, batch_size, img_size = canvases.shape
    y_recons = 1.0 / (1.0 + np.exp(-canvases))  # x_recons = sigmoid(canvas)
    print(y_recons.shape)
    B = A = int(np.sqrt(img_size / 3))
    # prefix = './output/myattn_deploy3_withoutatten'
    prefix = os.path.join(FLAGS.data_dir, 'myattn_deploy3_withoutatten')

    xtrain = xtrain[0:10,:];
    ytrain = ytrain [0:10, :];
    intensity_multiplier = 255  # 255

    for t in range(T):
        #img = xrecons_grid(X[t, :, :], B, A)

        img = np.zeros((2*B + 10 + 2*B,xtrain.shape[0]*A, 3))
        print(img.shape)
        for i in range(xtrain.shape[0]):
            # print(A,B,A*B*3)
            img[0:B, i * A:(i + 1) * A, :] = np.reshape(ytrain[i, :], (B, A, 3))*intensity_multiplier;
            img[B+4:2*B+4, i * A:(i + 1) * A, :] = np.reshape(xtrain[i, :], (B, A, 3))*intensity_multiplier;

            im = (np.reshape(y_recons[t, i, :], (B, A, 3)) * intensity_multiplier).astype(np.uint8)
            if(is_result_sharpen):
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                im = cv2.filter2D(im, -1, kernel)

            img[2 * B + 10:3 * B + 10, i * A:(i + 1) * A, :] = im;
            img[3*B+10:4*B+10, i * A:(i + 1) * A, :] = (np.reshape(xtrain[i, :], (B, A, 3)) * intensity_multiplier).astype(np.uint8);

            inp_im = np.reshape(ytrain[i, :], (B, A, 3))*intensity_multiplier;
            inp_name = os.path.join(FLAGS.data_dir, 'inp_%d.png' % i)
            out_name = os.path.join(FLAGS.data_dir, 'out_%d.png' % i)
            cv2.imwrite(inp_name, inp_im)
            cv2.imwrite(out_name, im)

        # plt.matshow(img, cmap=plt.cm.hot)
        # you can merge using imagemagick, i.e. convert -delay
        # 10 -loop 0 *.png mnist.gif

        imgname = '%s_%d.png' % (prefix, t)
        cv2.imwrite(imgname, img)
        #plt.savefig(imgname)
        print(imgname)
        plt.close()

    out_file = os.path.join(FLAGS.data_dir, "draw_data.npy")
    [Lxs, Lzs] = np.load(out_file)
    f = plt.figure()
    plt.plot(Lxs, label='Reconstruction Loss Lx')
    plt.plot(Lzs, label='Latent Loss Lz')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig('%s_loss.png' % (prefix))
