from tensorflow.examples.tutorials import mnist
import os
import numpy as np
import cv2
import sys
import time
import matplotlib
import constants as const
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BatchGenerator:

    def __init__(self,batch_size,read_dir=None):
        self.batch_size = batch_size;
        self.mnist = False;
        if(self.mnist):
            data_directory = os.path.join(read_dir, "mnist")
            if not os.path.exists(data_directory):
                os.makedirs(data_directory)
            self.train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data
        else:
            self.img_list = [];
            for i in range(5):
                img_path = './texture/simple/'+str(i)+'.jpg';
                img = cv2.imread(img_path,0)
                img = img.astype(np.float32) / 255

                self.img_list.append(img)

    def next(self,direction=const.Direction.UP,debug = False):
        if(self.mnist):
            xtrain,_=self.train_data.next_batch(self.batch_size) # xtrain is (batch_size x img_size)
        else:
            img_dims = const.A;
            xtrain = np.zeros((self.batch_size,img_dims  * img_dims ),dtype=np.float32)
            ytrain = np.zeros((self.batch_size, img_dims * img_dims), dtype=np.float32)
            if(debug):
                np.random.seed(10);

            ran_imgs = np.random.randint(len(self.img_list), size=self.batch_size);

            pts = np.random.rand(self.batch_size,2)
            #pts = np.ones((self.batch_size,2))* 0.2;
            #print('direction ', direction, const.Direction.UP.value)
            for i in range(self.batch_size):
                current_img = self.img_list[ran_imgs[i]]

                row,col = img_dims + pts[i,0:2] * np.subtract(current_img.shape,(3*img_dims,3*img_dims))
                row,col = int(row),int(col)

                #print(row,col,current_img.shape)

                if(row + img_dims > current_img.shape[0]):
                    print('Something is bad');

                if (col + img_dims > current_img.shape[1]):
                    print('Something is bad');

                img_x = current_img[row:row + img_dims, col:col + img_dims];

                if direction == const.Direction.UP.value:
                    img_y = current_img[row- img_dims:row , col:col + img_dims];
                elif direction == const.Direction.DOWN.value:
                    img_y = current_img[row + img_dims:row + 2* img_dims, col:col + img_dims];
                    #print(np.subtract(current_img.shape, (2 * img_dims, 2 * img_dims)))
                    #print(row + img_dims, row + 2* img_dims)
                    #print(row,col,current_img.shape)
                    #print(img_y.shape)
                elif direction == const.Direction.RIGHT.value:
                    img_y = current_img[row:row + img_dims, col+ img_dims :col + 2*img_dims];
                elif direction == const.Direction.LEFT.value:
                    img_y = current_img[row:row + img_dims, col - img_dims:col];

                xtrain[i,:] = img_x.flatten();
                ytrain[i, :] = img_y.flatten();
                #print('Generating texture');
        return xtrain,xtrain


if __name__ == '__main__':
    batch_gen = BatchGenerator(100,'./mnist')
    #batch_gen = BatchGenerator(100)
    start_time = time.time()
    x,y = batch_gen.next();
    elapsed_time = time.time() - start_time
    print('Time to generate batch ',elapsed_time)
    print(x.shape);
    img = x[10,:];
    img = np.reshape(img,(const.A,const.B));
    print(img.dtype)
    print(np.unique(img))
    plt.imshow(img)
    plt.savefig('./output/sub_texture.png')
