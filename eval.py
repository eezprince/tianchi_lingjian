from model import Model
import tensorflow as tf
import utils


session = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
config = utils.parse_config("data/config.yaml")
config["training_data"] += "_test"
print(config)
model = Model(config, False)
saver = tf.train.Saver()
saver.restore(session, tf.train.latest_checkpoint(model.output_dir))

length = len(Model.data_inputs)
bs = config["batch_size"]
real_outputs = list()
for start in xrange(0, length, bs):
    end = min(length, start + bs)
    batch_input = model.data_inputs[start:end]
    if end - start < bs:
        batch_input += batch_input[-1] * (bs - end + start)
    outputs = session.run(model.outputs_hat,
                    feed_dict={
                        model.inputs.name:batch_input,
                        model.is_training.name:False
                    })
    real_outputs.extend(outputs[:end-start])

# denormalize
with open(r"data/schema.txt", "r") as f_in:
    mean, std = f_in.readlines()[-1].split('\t')
with open(r"data/testA.txt", "r") as f_in:
    ids = [line.split("\t")[0] for line in f_in.readlines()]
with open(r"data/output/output.txt", "w") as f_out:
    for i, out in enumerate(real_outputs):
        f_out.write("{},{}\n".format(ids[i+1], out * float(std) + float(mean)))

