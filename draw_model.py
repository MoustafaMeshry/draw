import tensorflow as tf
import inspect
import constants as const



class DrawModel:

    '''
    Public variables are
    x: input
    y: output
    cs: hidden output
    Lz: hidden output distribution
    '''

    def __init__(self,read_attn,write_attn):
        self._DO_SHARE = None  # workaround for variable_scope(reuse = True)
        self._read_size = 2 * const.read_n * const.read_n if read_attn else 2 * const.img_size
        self._write_size = const.write_n * const.write_n if write_attn else const.img_size
        self._batch_size = const.batch_size
        DO_SHARE = None  # workaround for variable_scope(reuse = True)

        # x is our input (batch_size * img_size)
        self.x = tf.placeholder(tf.float32, shape=(const.batch_size, const.img_size))
        self.y = tf.placeholder(tf.float32, shape=(const.batch_size, const.img_size))
        self._e = tf.random_normal((const.batch_size, const.z_size), mean=0, stddev=1)  # Qsampler noise
        self._lstm_enc = tf.contrib.rnn.LSTMCell(const.enc_size, state_is_tuple=True)  # encoder Op
        self._lstm_dec = tf.contrib.rnn.LSTMCell(const.dec_size, state_is_tuple=True)  # decoder Op
        read = self.read_attn if read_attn else self.read_no_attn
        write = self.write_attn if write_attn else self.write_no_attn

        # ==STATE VARIABLES== #

        self.cs = [0] * const.T  # sequence of canvases

        # gaussian params generated by SampleQ. We will need these for computing loss.
        mus, logsigmas, sigmas = [0] * const.T, [0] * const.T, [0] * const.T
        # initial states
        h_dec_prev = tf.zeros((const.batch_size, const.dec_size))
        enc_state = self._lstm_enc.zero_state(const.batch_size, tf.float32)
        dec_state = self._lstm_dec.zero_state(const.batch_size, tf.float32)

        # ==DRAW MODEL== #

        # construct the unrolled computational graph
        for t in range(const.T):
            c_prev = tf.zeros((const.batch_size, const.img_size)) if t == 0 else self.cs[t - 1]
            # c_prev = tf.truncated_normal((const.batch_size, const.img_size), mean=.5, stddev=.1) if t == 0 else self.cs[t - 1]
            x_hat = self.y - tf.sigmoid(c_prev)  # error image
            # FIXME: be careful about whether y and c_prev are in the same range or not (e.g. normalized or not)
            # x_hat = self.y - c_prev  # error image
            r = read(self.x, x_hat, h_dec_prev)
            h_enc, enc_state = self.encode(enc_state, tf.concat([r, h_dec_prev], 1))
            z, mus[t], logsigmas[t], sigmas[t] = self.sampleQ(h_enc)
            h_dec, dec_state = self.decode(dec_state, z)
            self.cs[t] = c_prev + write(h_dec)  # store results
            h_dec_prev = h_dec
            self._DO_SHARE = True  # from now on, share variables



        kl_terms = [0] * const.T
        for t in range(const.T):
            mu2 = tf.square(mus[t])
            sigma2 = tf.square(sigmas[t])
            logsigma = logsigmas[t]

            # each kl term is (1xminibatch)
            kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - .5

        # this is 1xminibatch, corresponding to summing kl_terms from 1:T
        KL = tf.add_n(kl_terms)
        self.Lz = tf.reduce_mean(KL)  # average over minibatches



    def linear(self,x_var, output_dim):
        """
        affine transformation Wx+b
        assumes x.shape = (batch_size, num_features)
        """
        w = tf.get_variable("w", [x_var.get_shape()[1], output_dim])
        b = tf.get_variable("b", [output_dim],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x_var, w) + b

    def filterbank(self,gx, gy, sigma2, delta, N):
        grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta  # eq 19
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta  # eq 20
        a = tf.reshape(tf.cast(tf.range(const.A), tf.float32), [1, 1, -1])
        b = tf.reshape(tf.cast(tf.range(const.B), tf.float32), [1, 1, -1])
        mu_x = tf.reshape(mu_x, [-1, N, 1])
        mu_y = tf.reshape(mu_y, [-1, N, 1])
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square((a - mu_x) / (2 * sigma2)))  # 2*sigma2?
        Fy = tf.exp(-tf.square((b - mu_y) / (2 * sigma2)))  # batch x N x B
        # normalize, sum over A and B dims
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), const.eps)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), const.eps)
        return Fx, Fy

    def attn_window(self,scope, h_dec, N):
        with tf.variable_scope(scope, reuse=self._DO_SHARE):
            params = self.linear(h_dec, 5)
        # gx_, gy_, log_sigma2, log_delta, log_gamma=tf.split(1, 5, params)
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(params, 5, 1)
        gx = (const.A + 1) / 2 * (gx_ + 1)
        gy = (const.B + 1) / 2 * (gy_ + 1)
        sigma2 = tf.exp(log_sigma2)
        delta = (max(const.A, const.B) - 1) / (N - 1) * tf.exp(log_delta)  # batch x N
        return self.filterbank(gx, gy, sigma2, delta, N) + (tf.exp(log_gamma),)

    # ==READ== #
    def read_no_attn(self,x, x_hat, h_dec_prev):
        return tf.concat([x, x_hat], 1)

    def read_attn(self,x, x_hat, h_dec_prev):
        Fx, Fy, gamma = self.attn_window("read", h_dec_prev, const.read_n)

        def filter_img(img, Fx, Fy, gamma, N):
            Fxt = tf.transpose(Fx, perm=[0, 2, 1])
            img = tf.reshape(img, [-1, const.B, const.A])
            glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))
            glimpse = tf.reshape(glimpse, [-1, N * N])
            return glimpse * tf.reshape(gamma, [-1, 1])

        x = filter_img(x, Fx, Fy, gamma, const.read_n)  # batch x (read_n*read_n)
        x_hat = filter_img(x_hat, Fx, Fy, gamma, const.read_n)
        return tf.concat([x, x_hat], 1)  # concat along feature axis



    # ==ENCODE== #
    def encode(self,state, input):
        """
        run LSTM
        state = previous encoder state
        input = cat(read, h_dec_prev)
        returns: (output,  new_state)
        """
        with tf.variable_scope("encoder", reuse=self._DO_SHARE):
            return self._lstm_enc(input, state)

    # ==Q-SAMPLER (VARIATIONAL AUTOENCODER)== #

    def sampleQ(self,h_enc):
        """
        Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
        mu is (batch, z_size)
        """
        with tf.variable_scope("mu", reuse=self._DO_SHARE):
            mu = self.linear(h_enc, const.z_size)
        with tf.variable_scope("sigma", reuse=self._DO_SHARE):
            logsigma = self.linear(h_enc, const.z_size)
            sigma = tf.exp(logsigma)
        return (mu + sigma * self._e, mu, logsigma, sigma)

    # ==DECODER== #
    def decode(self,state, input):
        with tf.variable_scope("decoder", reuse=self._DO_SHARE):
            return self._lstm_dec(input, state)

    # ==WRITER== #
    def write_no_attn(self,h_dec):
        with tf.variable_scope("write", reuse=self._DO_SHARE):
            return self.linear(h_dec, const.img_size)

    def write_attn(self,h_dec):
        with tf.variable_scope("writeW", reuse=self._DO_SHARE):
            w = self.linear(h_dec, self._write_size)  # batch x (write_n*write_n)
        N = const.write_n
        w = tf.reshape(w, [self._batch_size, N, N])
        Fx, Fy, gamma = self.attn_window("write", h_dec, const.write_n)
        Fyt = tf.transpose(Fy, perm=[0, 2, 1])
        wr = tf.matmul(Fyt, tf.matmul(w, Fx))
        wr = tf.reshape(wr, [self._batch_size, const.B * const.A])
        # gamma = tf.tile(gamma, [1, B*A])
        return wr * tf.reshape(1.0 / gamma, [-1, 1])







