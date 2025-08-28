import tensorflow as tf

# CSI metric
class CriticalSuccessIndex(tf.keras.metrics.Metric):

    def __init__(self, threshold, name='csi', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.num_tp = self.add_weight(name='tp', initializer='zeros')
        self.num_tp_fp_fn = self.add_weight(name='tp_fp_fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.math.greater_equal(y_true, self.threshold)
        y_pred_bin = tf.math.greater_equal(y_pred, self.threshold)

        tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true_bin, y_pred_bin), tf.float32))
        tp_fp_fn = tf.reduce_sum(tf.cast(tf.math.logical_or(y_true_bin, y_pred_bin), tf.float32))

        self.num_tp.assign_add(tp)
        self.num_tp_fp_fn.assign_add(tp_fp_fn)

    def result(self):
        return tf.math.divide_no_nan(self.num_tp, self.num_tp_fp_fn) # returns 0 if denominator is zero

    def reset_states(self):
        self.num_tp.assign(0.0)
        self.num_tp_fp_fn.assign(0.0)


# POD metric
class ProbabilityOfDetection(tf.keras.metrics.Metric):

    def __init__(self, threshold, name='pod', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.num_tp = self.add_weight(name='tp', initializer='zeros')
        self.num_tp_fn = self.add_weight(name='tp_fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.math.greater_equal(y_true, self.threshold)
        y_pred_bin = tf.math.greater_equal(y_pred, self.threshold)

        tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true_bin, y_pred_bin), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true_bin, tf.math.logical_not(y_pred_bin)), tf.float32))

        self.num_tp.assign_add(tp)
        self.num_tp_fn.assign_add((tp + fn))

    def result(self):
        return tf.math.divide_no_nan(self.num_tp, self.num_tp_fn)

    def reset_states(self):
        self.num_tp.assign(0.0)
        self.num_tp_fn.assign(0.0)


# FAR metric
class FalseAlarmRatio(tf.keras.metrics.Metric):

    def __init__(self, threshold, name='far', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.num_fp = self.add_weight(name='fp', initializer='zeros')
        self.num_tp_fp = self.add_weight(name='tp_fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.math.greater_equal(y_true, self.threshold)
        y_pred_bin = tf.math.greater_equal(y_pred, self.threshold)

        tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true_bin, y_pred_bin), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true_bin), y_pred_bin), tf.float32))

        self.num_fp.assign_add(fp)
        self.num_tp_fp.assign_add((tp + fp))

    def result(self):
        return tf.math.divide_no_nan(self.num_fp, self.num_tp_fp)

    def reset_states(self):
        self.num_fp.assign(0.0)
        self.num_tp_fp.assign(0.0)


# FBI metric
class FrequencyBiasIndex(tf.keras.metrics.Metric):

    def __init__(self, threshold, name='fbi', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.num_tp_fp = self.add_weight(name='tp_fp', initializer='zeros')
        self.num_tp_fn = self.add_weight(name='tp_fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.math.greater_equal(y_true, self.threshold)
        y_pred_bin = tf.math.greater_equal(y_pred, self.threshold)

        tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true_bin, y_pred_bin), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true_bin), y_pred_bin), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true_bin, tf.math.logical_not(y_pred_bin)), tf.float32))

        self.num_tp_fp.assign_add((tp + fp))
        self.num_tp_fn.assign_add((tp + fn))

    def result(self):
        return tf.math.divide_no_nan(self.num_tp_fp, self.num_tp_fn)

    def reset_states(self):
        self.num_tp_fp.assign(0.0)
        self.num_tp_fn.assign(0.0)


# ACC metric
class ACC(tf.keras.metrics.Metric):

    def __init__(self, threshold, name='acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.num_tp_tn = self.add_weight(name='tp_tn', initializer='zeros')
        self.num_total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.math.greater_equal(y_true, self.threshold)
        y_pred_bin = tf.math.greater_equal(y_pred, self.threshold)

        tp_tn = tf.reduce_sum(tf.cast(tf.math.equal(y_true_bin, y_pred_bin), tf.float32))
        total = tf.cast(tf.size(y_true_bin), tf.float32)

        self.num_tp_tn.assign_add(tp_tn)
        self.num_total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.num_tp_tn, self.num_total)

    def reset_states(self):
        self.num_tp_tn.assign(0.0)
        self.num_total.assign(0.0)


# HSS metric
class HeidkeSkillScore(tf.keras.metrics.Metric):

    def __init__(self, threshold, name='hss', **kwargs):# 여기
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.num_tp_fp = self.add_weight(name='tp_fp', initializer='zeros')
        self.num_tp_fn = self.add_weight(name='tp_fn', initializer='zeros')
        self.num_fp_tn = self.add_weight(name='fp_tn', initializer='zeros')
        self.num_fn_tn = self.add_weight(name='fn_tn', initializer='zeros')
        self.num_tp_tn = self.add_weight(name='tp_tn', initializer='zeros')
        self.num_total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.math.greater_equal(y_true, self.threshold)
        y_pred_bin = tf.math.greater_equal(y_pred, self.threshold)

        tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true_bin, y_pred_bin), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true_bin), y_pred_bin), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true_bin, tf.math.logical_not(y_pred_bin)), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.logical_not(tf.math.logical_or(y_true_bin, y_pred_bin)), tf.float32))
        total = tf.cast(tf.size(y_true_bin), tf.float32)

        self.num_tp_fp.assign_add((tp + fp))
        self.num_tp_fn.assign_add((tp + fn))
        self.num_fp_tn.assign_add((fp + tn))
        self.num_fn_tn.assign_add((fn + tn))
        self.num_tp_tn.assign_add((tp + tn))
        self.num_total.assign_add(total)

    def result(self):
        e = tf.math.divide_no_nan((((self.num_tp_fp) * (self.num_tp_fn)) + ((self.num_fp_tn) * (self.num_fn_tn))), self.num_total)
        return tf.math.divide_no_nan((self.num_tp_tn - e), (self.num_total - e))

    def reset_states(self):
        self.num_tp_fp.assign(0.0)
        self.num_tp_fn.assign(0.0)
        self.num_fp_tn.assign(0.0)
        self.num_fn_tn.assign(0.0)
        self.num_tp_tn.assign(0.0)
        self.num_total.assign(0.0)


# TP, FP, FN, TN
class FactorsOfConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, threshold, name='factors_of_confusion_matrix', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.num_tp = self.add_weight(name='tp', initializer='zeros')
        self.num_fp = self.add_weight(name='fp', initializer='zeros')
        self.num_fn = self.add_weight(name='fn', initializer='zeros')
        self.num_tn = self.add_weight(name='tn', initializer='zeros')
        self.num_total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.math.greater_equal(y_true, self.threshold)
        y_pred_bin = tf.math.greater_equal(y_pred, self.threshold)

        tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true_bin, y_pred_bin), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true_bin), y_pred_bin), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true_bin, tf.math.logical_not(y_pred_bin)), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.logical_not(tf.math.logical_or(y_true_bin, y_pred_bin)), tf.float32))
        total = tf.cast(tf.size(y_true_bin), tf.float32)

        self.num_tp.assign_add(tp)
        self.num_fp.assign_add(fp)
        self.num_fn.assign_add(fn)
        self.num_tn.assign_add(tn)
        self.num_total.assign_add(total)

    def result(self):
        return {
            'TP': self.num_tp,
            'FP': self.num_fp,
            'FN': self.num_fn,
            'TN': self.num_tn,
            'TOTAL': self.num_total
        }

    def reset_states(self):
        self.num_tp.assign(0.0)
        self.num_fp.assign(0.0)
        self.num_fn.assign(0.0)
        self.num_tn.assign(0.0)
        self.num_total.assign(0.0)