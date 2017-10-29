from tensorflow.examples.tutorials import mnist
import os

class BatchGenerator:
    def __init__(self,batch_size,read_dir=None):
        self.batch_size = batch_size;
        #print(read_dir)
        if(read_dir is not None):
            data_directory = os.path.join(read_dir, "mnist")
            if not os.path.exists(data_directory):
                os.makedirs(data_directory)
            self.train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data
    def next(self):
        xtrain,_=self.train_data.next_batch(self.batch_size) # xtrain is (batch_size x img_size)
        #print(xtrain.shape);
        return xtrain,xtrain
    