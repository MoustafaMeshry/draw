attention_flag = False
batch_size = 100 # TODO for vgg_loss, this must be 1 (at least for now)
A, B = 28, 28 # image width, height
# A, B = 50, 50 # image width, height
# A, B = 80, 80 # image width, height
img_size = B*A*3  # the canvas size
enc_size = 256  # number of hidden units / output size in LSTM of the encoder
dec_size = 256  # number of hidden units / output size in LSTM of the decoder
read_n = 5  # read glimpse grid width/height
write_n = 5  # write glimpse grid width/height
z_size = 10  # QSampler output size
T = 10  # MNIST generation sequence length
learning_rate = 1e-3  # learning rate for optimizer
eps = 1e-8  # epsilon for numerical stability

vgg_model_path = '/fs/vulcan-scratch/mmeshry/DRAW/lib/weights/vgg19.npy'

gpu_used = True
if(gpu_used):
    train_iters = 10000
else:
    train_iters = 1100

from enum import Enum
class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

