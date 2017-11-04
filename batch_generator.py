from tensorflow.examples.tutorials import mnist
import os
import numpy as np
import cv2
import sys
import time
import matplotlib
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

    def next(self):
        if(self.mnist):
            xtrain,_=self.train_data.next_batch(self.batch_size) # xtrain is (batch_size x img_size)
        else:
            img_dims = 28;
            xtrain = np.zeros((self.batch_size,img_dims  * img_dims ),dtype=np.float32)
            ran_imgs = np.random.randint(len(self.img_list), size=self.batch_size);
            pts = np.random.rand(self.batch_size,2)
            for i in range(self.batch_size):
                current_img = self.img_list[ran_imgs[i]]

                row,col = pts[i,0:2] * current_img.shape
                row,col = int(row),int(col)
                #print(row,col,current_img.shape)

                if(row + img_dims > current_img.shape[0]):
                    row = current_img.shape[0] - img_dims;

                if (col + img_dims > current_img.shape[1]):
                    col = current_img.shape[1] - img_dims;

                xtrain[i,:] = current_img[row:row+img_dims,col:col+img_dims].flatten();
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
    img = np.reshape(img,(28,28));
    print(img.dtype)
    print(np.unique(img))
    plt.imshow(img)
    plt.savefig('./dummy/sub_texture.png')
