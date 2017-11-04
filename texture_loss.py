import tensorflow as tf
import constants as const

class TextureLoss:
    # ==LOSS FUNCTION== #
    def binary_crossentropy(self,t, o):
        return -(t * tf.log(o + const.eps) + (1.0 - t) * tf.log(1.0 - o + const.eps))

    def texture_filter_bank_loss(self,t, o):
        return self.binary_crossentropy(t, o);
