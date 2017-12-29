import tensorflow as tf
from model import Model
import os
import time
import utils


def train(config):
    train_model = Model(config, True)
    test_model = Model(config, False)

    if tf.gfile.Exists(train_model.output_dir):
        tf.gfile.DeleteRecursively(train_model.output_dir)
        tf.gfile.MakeDirs(train_model.output_dir)

    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    summary_writer = tf.summary.FileWriter(train_model.output_dir, session.graph)

    session.run(tf.global_variables_initializer())

    start_time = time.time()
    for step in xrange(train_model.config["max_steps"] + 1):
        feed_dict = dict()
        batch = train_model.get_batch_data(True)
        feed_dict[train_model.inputs.name] = batch[0]
        feed_dict[train_model.outputs.name] = batch[1]
        _, loss_value = session.run([train_model.train_op, train_model.cost], feed_dict=feed_dict)

        duration = time.time() - start_time
        if step % 100 == 0:
            print('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))
        if step % 1000 == 0:
            saver.save(session, os.path.join(train_model.output_dir, r"model"), global_step=step)
            if test_model.test_size != 0:
                val_batch = test_model.get_batch_data(False)
                feed_dict[test_model.inputs.name] = val_batch[0]
                feed_dict[test_model.outputs.name] = val_batch[1]
                val_loss_value = session.run(test_model.cost, feed_dict=feed_dict)
                duration = time.time() - start_time
                print('Step %d: val_loss = %.4f (%.3f sec)' % (step, val_loss_value, duration))


def main():
    config = utils.parse_config("data/config.yaml")
    print(config)
    train(config)

if __name__ == '__main__':
    print("training start")
    with tf.Graph().as_default():
        main()
    print("training finish")
