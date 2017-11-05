batch_size = 100  # training minibatch size
A, B = 28, 28  # image width, height
img_size = B*A  # the canvas size
enc_size = 256  # number of hidden units / output size in LSTM of the encoder
dec_size = 256  # number of hidden units / output size in LSTM of the decoder
read_n = 5  # read glimpse grid width/height
write_n = 5  # write glimpse grid width/height
z_size = 10  # QSampler output size
T = 10  # MNIST generation sequence length
learning_rate = 1e-3  # learning rate for optimizer
eps = 1e-8  # epsilon for numerical stability

gpu_used = False
if(gpu_used):
    gpu_train_iters = 10000
else:
    cpu_train_iters = 10000

train_iters = cpu_train_iters


