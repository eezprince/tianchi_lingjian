import tensorflow as tf
from model import Model
import os
import time
import utils


def add_summary(writer, name, value, step):
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=value)
    writer.add_summary(summary, step)

def train(config):
    model = Model(config)

    if tf.gfile.Exists(model.output_dir):
        tf.gfile.DeleteRecursively(model.output_dir)
        tf.gfile.MakeDirs(model.output_dir)

    saver = tf.train.Saver()
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model.output_dir, session.graph)

    session.run(tf.global_variables_initializer())

    avg_loss = 0
    for step in xrange(model.config["max_steps"] + 1):
        feed_dict = dict()
        batch = model.get_batch_data(True)
        feed_dict[model.inputs.name] = batch[0]
        feed_dict[model.outputs.name] = batch[1]
        feed_dict[model.is_training.name] = True
        _, loss_value = session.run([model.train_op, model.cost], feed_dict=feed_dict)
        avg_loss += loss_value
        if step % 100 == 0 and step != 0:
            print('Step %d: loss = %.4f' % (step, avg_loss/100))
            avg_loss=0
            add_summary(summary_writer, "training_loss", loss_value, step)
        if step % 1000 == 0:
            saver.save(session, os.path.join(model.output_dir, r"model"), global_step=step)
            if model.test_size != 0:
                val_batch = model.get_batch_data(False)
                feed_dict[model.inputs.name] = val_batch[0]
                feed_dict[model.outputs.name] = val_batch[1]
                feed_dict[model.is_training.name] = False
                val_loss_value, lr = session.run([model.cost, model.learning_rate], feed_dict=feed_dict)
                print('Step %d: val_loss = %.4f, lr = %.6f' % (step, val_loss_value, lr))
                add_summary(summary_writer, "val_loss", val_loss_value, step)
                add_summary(summary_writer, "learningrate", lr, step)
                summary_writer.flush()


def main():
    config = utils.parse_config("data/config.yaml")
    print(config)
    train(config)

if __name__ == '__main__':
    print("training start")
    with tf.Graph().as_default():
        main()
    print("training finish")
