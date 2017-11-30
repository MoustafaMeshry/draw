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

    direction = const.Direction.UP.value
    size = const.A
    with_attention = False
    is_result_sharpen = False




    B = A = const.A

    prefix = './output/myattn_deploy3'

    t = 9 ## Final Layer
    img = np.zeros((A*3, B*3));
    pred_img = np.zeros((A * 3, B * 3));
    model = draw_model.DrawModel(with_attention, with_attention);
    for texture_id in range(10):
        for dir in range(4):
            save_path = os.path.join("./train/",
                                     'simple_d' + str(dir) + '_s' + str(size) + '_a' + str(with_attention));
            #save_path = os.path.join("./train/",
            #                         'tmp_simple_d' + str(direction) + '_s' + str(size) + '_a' + str(with_attention));

            FLAGS = tf.flags.FLAGS
            img_generator = batch_gen.BatchGenerator(const.batch_size, save_path)


            sess = tf.InteractiveSession()
            saver = tf.train.Saver()  # saves variables learned during training
            tf.global_variables_initializer().run()

            ckpt_file = os.path.join(save_path , "drawmodel.ckpt")
            saver.restore(sess, ckpt_file)


            xtrain, ytrain = img_generator.next(direction=dir, debug=True)

            feed_dict = {model.x: xtrain, model.y: ytrain}
            canvases = sess.run(model.cs, feed_dict)  # generate some examples
            canvases = np.array(canvases)  # T x batch x img_size

            # visualize results
            T, batch_size, img_size = canvases.shape
            y_recons = 1.0 / (1.0 + np.exp(-canvases))  # x_recons = sigmoid(canvas)
            print(y_recons.shape)

            xtrain = xtrain[texture_id , :]*255;
            ytrain = ytrain[texture_id , :]*255;
            y_recons = y_recons[9,texture_id ,:]*255

            if(dir == const.Direction.UP.value):
                img[0:A,A:2*A] = np.reshape(ytrain,(A,B))
                pred_img[0:A, A:2 * A] = np.reshape(y_recons , (A, B))
            elif(dir == const.Direction.DOWN.value):
                img[2*A:3*A, A:2 * A] = np.reshape(ytrain,(A,B))
                pred_img[2 * A:3 * A, A:2 * A] = np.reshape(y_recons , (A, B))
            elif (dir == const.Direction.LEFT.value):
                img[A:2*A, 0:A] = np.reshape(ytrain,(A,B))
                pred_img[A:2 * A, 0:A] = np.reshape(y_recons , (A, B))
            elif (dir == const.Direction.RIGHT.value):
                img[A:2 * A, 2*A:3*A] = np.reshape(ytrain,(A,B))
                pred_img[A:2 * A, 2 * A:3 * A] = np.reshape(y_recons , (A, B))

            img[A:A*2,A:A*2]= np.reshape(xtrain,(A,B))
            pred_img[A:A * 2, A:A * 2] = np.reshape(xtrain, (A, B))

        cv2.imwrite('./output/myattn_deploy3_'+str(texture_id)+'real.png', img)
        cv2.imwrite('./output/myattn_deploy3_'+str(texture_id)+'predicted.png', pred_img)

