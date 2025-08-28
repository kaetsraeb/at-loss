import tensorflow as tf

"""
2025 AT loss - with annealing
"""
class NewAdvancedTorrentialLoss(tf.keras.losses.Loss):
    def __init__(self, threshold, call_per_epoch, tau_init = 1.0, tau_decay = 0.005, tau_min = 0.05, scale = 0.05,
                 reduction=tf.keras.losses.Reduction.NONE, name='advanced_torrential_loss'):
        super(NewAdvancedTorrentialLoss, self).__init__(reduction=reduction, name=name)
        self.threshold = threshold
        self.call_per_epoch = tf.constant(int(call_per_epoch), dtype=tf.int64)
        self.tau_init = tau_init
        self.tau_decay = tau_decay
        self.tau_min = tau_min
        self.scale = scale
        self.count = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.num_decay = tf.Variable(0, trainable=False, dtype=tf.float32)

    def call(self, y_true, y_pred):

        # (self.count % self.call_per_epoch) == 0, then self.num_decay = self.num_decay + 1.0
        condition = tf.math.equal(x=tf.math.floormod(x=self.count, y=self.call_per_epoch), y=0)
        tf.cond(pred=condition, true_fn=lambda: self.num_decay.assign_add(delta=1.0), false_fn=lambda: self.num_decay)
 
        # annealed temperature
        tau = tf.math.maximum(x=(self.tau_init - (self.tau_decay * (self.num_decay - 1.0))), y=self.tau_min)

        # self.count = self.count + 1
        self.count.assign_add(delta=1)

        # z_true
        z_true = tf.where(condition=(y_true >= self.threshold), x=1.0, y=0.0) # true condition element -> 1.0, false condition element -> 0.0

        # z_pred
        l = self._sampling_logistic_noise(tf.shape(y_pred))
        z_pred = tf.math.sigmoid(x=(((2 * y_pred) - (2 * self.threshold) + l) / tau))

        # cost return
        z = tf.math.square(x=(z_true - z_pred)) # penalty matrix [0, 1]
        cost = tf.reduce_mean(input_tensor=z)
        return cost 

    def _sampling_logistic_noise(self, shape):
        seed_value = tf.cast(self.num_decay.read_value(), dtype=tf.int32)
        seed_tensor = tf.stack([seed_value, (seed_value + 1)])
        u = tf.random.stateless_uniform(shape=shape, minval=1e-10, maxval=1.0, seed=seed_tensor)
        logistic = tf.math.log(x=u) - tf.math.log(x=(1.0 - u)) # L ~ Logistic(0, 1)
        return logistic * self.scale

