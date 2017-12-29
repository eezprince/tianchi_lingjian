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
        self._load_data()
        dropout_rate = self.dropout_rate
        reuse = False
        if not is_training:
            reuse = True
        with tf.variable_scope("", reuse=reuse):
            self.inputs = tf.placeholder(tf.float32, shape=[self.batch_size, Model.input_dim], name="inputs")
            self.input_layer = self.forwardprop(self.inputs, Model.input_dim, self.hidden_dim, "input_layer")
            self.input_dropout_layer = tf.layers.dropout(self.input_layer, dropout_rate,
                    training=is_training, name="input_droput_layer")
            self.hidden_layer = self.forwardprop(self.input_dropout_layer, self.hidden_dim,
                    self.hidden_dim, "hidden_layer")
            self.hidden_dropout_layer = tf.layers.dropout(self.hidden_layer, dropout_rate,
                    training=is_training, name="hidden_dropout_layer")
            self.hidden2_layer = self.forwardprop(self.hidden_dropout_layer, self.hidden_dim,
                    self.hidden_dim, "hidden2_layer")
            self.outputs_hat = components.linear_layer(self.hidden2_layer, self.hidden_dim,
                    1, True, "output_layer")
            self.outputs = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name="outputs")
            self.cost = tf.reduce_mean(tf.square(self.outputs - self.outputs_hat))
            global_step = tf.Variable(0, name='global_step', trainable=False)

            learning_rate = tf.train.exponential_decay(self.lr, global_step, 3000, 0.96, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(self.cost, global_step=global_step)


    def forwardprop(self, inputs, input_dim, output_dim, variable_scope):
        linear_outputs = components.linear_layer(inputs, input_dim,
                output_dim, bias=True, variable_scope=variable_scope)
        return tf.nn.relu(linear_outputs)


    def _check_config(self, config):
        required_settings = ["log_dir", "learning_rate", "max_steps", "hidden_dim", "batch_size",
                "dropout_rate", "training_data", "test_size"]
        for setting in required_settings:
            if setting not in config:
                return setting
        return ""


    def _load_data(self):
        if not Model.data_loaded:
            with open(self.training_data, "r") as f_in:
                datas = [[float(x) for x in line.split("\t")] for line in f_in.read().splitlines()]
            Model.input_dim = len(datas[0]) - 1
            Model.data_inputs = [data[:-1] for data in datas]
            Model.data_outputs = [data[-1:] for data in datas]
            Model.data_train_inputs, Model.data_test_inputs, \
            Model.data_train_outputs, Model.data_test_outputs = \
                cross_validation.train_test_split(self.data_inputs, self.data_outputs,
                                                  test_size=self.test_size, random_state=0)


    def get_batch_data(self, training_data=True):
        if training_data:
            return zip(*random.sample(zip(Model.data_train_inputs, Model.data_train_outputs), self.batch_size))
        else:
            return zip(*random.sample(zip(Model.data_test_inputs, Model.data_test_outputs), self.batch_size))
