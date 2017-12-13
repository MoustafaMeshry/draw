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


def generateTile(output_img,radius,dir,texture_list=[0],input_img_col_shift=0):
    if(dir == const.Direction.RIGHT):
        col_offest = 1;
        row_offest = 0;
    elif(dir == const.Direction.LEFT):
        col_offest = -1;
        row_offest = 0;
    elif (dir == const.Direction.UP):
        col_offest = 0;
        row_offest = -1;
    elif (dir == const.Direction.DOWN):
        col_offest = 0;
        row_offest = 1;

    for idx,texture_id in enumerate(texture_list):
        for i in range(radius):
            in_offset = [const.B*(radius+i*row_offest),const.B*(radius+i*col_offest+input_img_col_shift)]
            out_offest = [const.B*(radius+(i+1)*row_offest),const.B*(radius+(i+1)*col_offest+input_img_col_shift)]

            in_img = output_img[in_offset[0]:in_offset[0]+const.B,in_offset[1]:in_offset[1]+const.B,idx];

            row_img = in_img.flatten();
            row_img = row_img [np.newaxis,:];
            row_img = np.repeat(row_img,100,0)
            ## Need to update the model to take variable batch size
            feed_dict = {model.x: row_img , model.y:row_img }
            canvases = sess.run(model.cs, feed_dict)  # generate some examples
            canvases = np.array(canvases)  # T x batch x img_size
            T, batch_size, img_size = canvases.shape
            y_recons = 1.0 / (1.0 + np.exp(-canvases))  # x_recons = sigmoid(canvas)
            #print(texture_id)
            #print(y_recons[-1,texture_id,:].shape)
            out_img = np.reshape(y_recons[-1,texture_id,:], (B, B))* 255
            #print(out_img.shape)
            #print(output_img[out_offest[0]:out_offest[0]+B,out_offest[1]:out_offest[1]+B,idx].shape)
            output_img[out_offest[0]:out_offest[0]+B,out_offest[1]:out_offest[1]+B,idx] = out_img;
            cv2.imwrite('./test/'+str(texture_id)+'myattn_deploy3_' + str(i) + 'real.png', output_img[:,:,idx])

if __name__ == '__main__':
    # load module

    dir = const.Direction.UP.value
    size = const.A
    with_attention = False
    is_result_sharpen = False
    B = A = const.A
    prefix = './output/myattn_deploy3'
    radius = 3;
    save_path = os.path.join("./train/",
                             'simple_d' + str(dir) + '_s' + str(size) + '_a' + str(with_attention));
    FLAGS = tf.flags.FLAGS
    img_generator = batch_gen.BatchGenerator(const.batch_size, save_path)

    xtrain, ytrain = img_generator.next(direction=dir, debug=True);

    texture_list = [0, 4, 10];
    input_img = np.reshape(np.transpose(xtrain[texture_list,:]),(A,B,len(texture_list)))

    output_img = np.zeros((B*(2*radius+1),B*(2*radius+1),len(texture_list)))

    output_img[B*radius:B*radius+B, B * radius:B * radius+B,:] = input_img * 255;
    #gt_img =
    ## Do Left to right

    model = draw_model.DrawModel(with_attention, with_attention);
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()  # saves variables learned during training
    tf.global_variables_initializer().run()




    ## Generate Right
    dir = const.Direction.RIGHT
    ckpt_file = os.path.join(save_path, "drawmodel.ckpt")  ## Should change to load the
    saver.restore(sess, ckpt_file)
    generateTile(output_img, radius, dir, texture_list);


    ## Generate Left
    dir = const.Direction.LEFT
    ckpt_file = os.path.join(save_path, "drawmodel.ckpt")  ## Should change to load the
    saver.restore(sess, ckpt_file)
    generateTile(output_img, radius, dir, texture_list);

    ## Generate Up

    dir = const.Direction.UP
    ckpt_file = os.path.join(save_path, "drawmodel.ckpt")  ## Should change to load the
    saver.restore(sess, ckpt_file)
    for i in range(-radius,radius + 1):
        generateTile(output_img, radius, dir, texture_list, i);

    ## Generate Down

    dir = const.Direction.DOWN
    ckpt_file = os.path.join(save_path, "drawmodel.ckpt")  ## Should change to load the
    saver.restore(sess, ckpt_file)
    for i in range(-radius,radius + 1):
        generateTile(output_img, radius, dir, texture_list, i);





    ''''
    t = 9 ## Final Layer
    img = np.zeros((A*3, B*3));
    pred_img = np.zeros((A * 3, B * 3));

    for texture_id in range(10):
        for dir in range(4):




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
    '''
