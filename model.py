import tensorflow as tf
import time
import components
import random
import numpy as np
from sklearn import cross_validation


class Model():

    data_loaded = False

    def __init__(self, config, is_training=True):
        config_error = self._check_config(config)
        assert config_error == "", "Config error {}.".format(config_error)
        self.config = config
        self.batch_size = config["batch_size"]
        self.hidden_dim = config["hidden_dim"]
        self.lr = self.config["learning_rate"]
        self.output_dir = self.config["log_dir"]
        self.dropout_rate = self.config["dropout_rate"]
        self.training_data = self.config["training_data"]
        self.test_size = self.config["test_size"]
        self.bn = True
        self.dropout = True
        self._load_data(is_training)
        dropout_rate = self.dropout_rate
        with tf.variable_scope("main"):
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.inputs = tf.placeholder(tf.float32, shape=[self.batch_size, Model.input_dim], name="inputs")
            hidden_layer = self.add_layer(self.inputs, Model.input_dim, self.hidden_dim, self.is_training,
                    self.dropout, self.dropout_rate, self.bn, "hidden_layer")
            self.outputs_hat = components.linear_layer(hidden_layer, self.hidden_dim,
                    1, True, "output_layer")
            self.outputs = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name="outputs")
            self.cost = tf.reduce_mean(tf.square(self.outputs - self.outputs_hat))
            regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = tf.identity(self.cost, "loss")
            global_step = tf.Variable(0, name='global_step', trainable=False)

            self.learning_rate = tf.train.exponential_decay(self.lr, global_step, 2000, 0.2, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.gvs = self.optimizer.compute_gradients(self.loss + 0.002*regularization_loss)

            # gradient clipping
            gradients = [grad for grad, var in self.gvs]
            params = [var for grad, var in self.gvs]
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, 1.0)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                #self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, params), global_step)
                 self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)


    def add_layer(self, inputs, input_dim, output_dim, is_training, dropout=False, dropout_rate=0,
            bn=False, variable_scope="dense_layer"):
        with tf.variable_scope(variable_scope):
            forward_out = self.forwardprop(inputs, input_dim, output_dim, is_training=is_training, bn=bn)
            if dropout:
                forward_out = tf.layers.dropout(forward_out, dropout_rate, training=is_training)
            return forward_out


    def forwardprop(self, inputs, input_dim, output_dim, is_training, bn=False, variable_scope="FF"):
        linear_outputs = components.linear_layer(inputs, input_dim, output_dim, bias=True)
        if bn:
            linear_outputs = tf.contrib.layers.batch_norm(linear_outputs, is_training=is_training, scope='bn')
        return tf.nn.relu(linear_outputs, "relu")


    def _check_config(self, config):
        required_settings = ["log_dir", "learning_rate", "max_steps", "hidden_dim", "batch_size",
                "dropout_rate", "training_data", "test_size"]
        for setting in required_settings:
            if setting not in config:
                return setting
        return ""


    def _load_data(self, is_training):
        if not Model.data_loaded:
            with open(self.training_data, "r") as f_in:
                datas = [[float(x) for x in line.split("\t")] for line in f_in.read().splitlines()]
            Model.input_dim = len(datas[0])
            if is_training:
                Model.input_dim -= 1
                Model.data_inputs = [data[:-1] for data in datas]
                Model.data_outputs = [data[-1:] for data in datas]
                Model.data_train_inputs, Model.data_test_inputs, \
                Model.data_train_outputs, Model.data_test_outputs = \
                    cross_validation.train_test_split(self.data_inputs, self.data_outputs,
                                                      test_size=self.test_size, random_state=0)
            else:
                Model.data_inputs = datas
            Model.data_loaded = True



    random.seed(0)
    def get_batch_data(self, training_data=True):
        if training_data:
            return zip(*random.sample(zip(Model.data_train_inputs, Model.data_train_outputs), self.batch_size))
        else:
            return zip(*random.sample(zip(Model.data_test_inputs, Model.data_test_outputs), self.batch_size))

