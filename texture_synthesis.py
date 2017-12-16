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
    generate_gif = False;
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

    gif_step = 0
    for idx,texture_id in enumerate(texture_list):
        for i in range(radius):
            in_offset = [const.B*(radius+i*row_offest),const.B*(radius+i*col_offest+input_img_col_shift)]

            out_offest = [const.B*(radius+(i+1)*row_offest),const.B*(radius+(i+1)*col_offest+input_img_col_shift)]

            in_img = output_img[in_offset[0]:in_offset[0]+const.B,in_offset[1]:in_offset[1]+const.B,:,idx];

            row_img = in_img.flatten();
            row_img = row_img [np.newaxis,:];
            row_img = np.repeat(row_img,const.batch_size,0)
            ## Need to update the model to take variable batch size
            feed_dict = {model.x: row_img }
            canvases = sess.run(model.cs, feed_dict)  # generate some examples
            canvases = np.array(canvases)  # T x batch x img_size

            y_recons = 1.0 / (1.0 + np.exp(-canvases))  # x_recons = sigmoid(canvas)

            out_img = np.reshape(y_recons[-1,texture_id,:], (B, B,const.num_channels))
            #cv2.imwrite('./test/out_real.png',out_img * 255)
            output_img[out_offest[0]:out_offest[0] + B, out_offest[1]:out_offest[1] + B, :, idx] = out_img;

            cv2.imwrite('./test/' + str(texture_id) + '_reconstructed.png',output_img[:, :, :, idx] * 255)
            if(generate_gif):
                cv2.imwrite('./test/'+str(generateTile.gif_step)+"_"+str(texture_id) + '_reconstructed.png', output_img[:,:,:,idx]* 255)
                generateTile.gif_step += 1;


if __name__ == '__main__':
    # load module
    generateTile.gif_step = 0
    dir = const.Direction.UP.value
    size = const.A
    with_attention = False
    is_result_sharpen = False
    B = A = const.A


    FLAGS = tf.flags.FLAGS
    img_generator = batch_gen.BatchGenerator(const.batch_size)

    xtrain, ytrain = img_generator.next(direction=dir, debug=True);

    radius = 3; # Tile around center
    texture_list = [3];

    input_img = np.reshape(np.transpose(xtrain[texture_list,:]),(A,B,const.num_channels,len(texture_list)))



    output_img = np.zeros((B*(2*radius+1),B*(2*radius+1),const.num_channels,len(texture_list)))
    output_img[B*radius:B*radius+B, B * radius:B * radius+B,:,:] = input_img;
    #gt_img =
    ## Do Left to right

    model = draw_model.DrawModel(with_attention, with_attention);
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()  # saves variables learned during training
    tf.global_variables_initializer().run()



    ## Generate Right
    dir = const.Direction.RIGHT
    save_path = os.path.join("./train/",
                             'simple_xx_rgb_d' + str(dir.value) + '_s' + str(size) + '_a' + str(with_attention));
    ckpt_file = os.path.join(save_path, "drawmodel.ckpt")  ## Should change to load the
    saver.restore(sess, ckpt_file)
    generateTile(output_img, radius, dir, texture_list);


    ## Generate Left
    dir = const.Direction.LEFT
    save_path = os.path.join("./train/",
                             'simple_xx_rgb_d' + str(dir.value) + '_s' + str(size) + '_a' + str(with_attention));
    ckpt_file = os.path.join(save_path, "drawmodel.ckpt")  ## Should change to load the
    saver.restore(sess, ckpt_file)
    generateTile(output_img, radius, dir, texture_list);

    ## Generate Up
    dir = const.Direction.UP
    save_path = os.path.join("./train/",
                             'simple_xx_rgb_d' + str(dir.value) + '_s' + str(size) + '_a' + str(with_attention));
    ckpt_file = os.path.join(save_path, "drawmodel.ckpt")  ## Should change to load the
    saver.restore(sess, ckpt_file)
    for i in range(-radius,radius + 1):
        generateTile(output_img, radius, dir, texture_list, i);

    ## Generate Down

    dir = const.Direction.DOWN
    save_path = os.path.join("./train/",
                             'simple_xx_rgb_d' + str(dir.value) + '_s' + str(size) + '_a' + str(with_attention));
    ckpt_file = os.path.join(save_path, "drawmodel.ckpt")  ## Should change to load the
    saver.restore(sess, ckpt_file)
    for i in range(-radius,radius + 1):
        generateTile(output_img, radius, dir, texture_list, i);


