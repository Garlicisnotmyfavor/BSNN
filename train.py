import tensorflow as tf
import time
import os
import datetime
from BSNN import BSNN
from DataUtil import read_dataset

#configuration
tf.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.flags.DEFINE_integer("num_epochs",10,"embedding size")
tf.flags.DEFINE_integer("batch_size", 30, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.flags.DEFINE_integer("num_classes", 3, "number of classes")
tf.flags.DEFINE_integer("vocab_size", 256, "vocabulary size")
tf.flags.DEFINE_integer("decay_steps", 12000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少

tf.flags.DEFINE_string("ckpt_dir","text_han_checkpoint/","checkpoint location for the model")
tf.flags.DEFINE_integer('num_checkpoints',5,'save checkpoints count')  

tf.flags.DEFINE_integer('max_sentence_num',100,'max sentence num in a doc')
tf.flags.DEFINE_integer('max_sentence_length',8,'max word count in a sentence')
tf.flags.DEFINE_integer("embedding_size",256,"embedding size")
tf.flags.DEFINE_integer('hidden_size',50,'cell output size')

tf.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")

tf.flags.DEFINE_integer("validate_every", 100, "Validate every validate_every epochs.") #每10轮做一次验证
tf.flags.DEFINE_float('validation_percentage',0.1,'validat data percentage in train data')

tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float('grad_clip',2.0,'grad_clip') # 和类别数相关

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS

# load dataset
train_x, train_y, dev_x, dev_y = read_dataset()
print ("data load finished")

with tf.Graph().as_default():
    sess_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=sess_conf)

    with sess.as_default():
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run_BSNN', timestamp))

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, FLAGS.ckpt_dir))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

        bsnn = BSNN(FLAGS.max_sentence_num, FLAGS.max_sentence_length, FLAGS.num_classes, FLAGS.vocab_size, FLAGS.embedding_size,
                  FLAGS.learning_rate, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.hidden_size, FLAGS.l2_reg_lambda,
                  FLAGS.grad_clip, FLAGS.is_training)
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {bsnn.input_x: x_batch,
                         bsnn.input_y: y_batch,
                         bsnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                         }
            tmp, step, loss, accuracy, label_cnt, pred_cnt, pred_min, pred_max = sess.run(
                [bsnn.train_op, bsnn.global_step, bsnn.loss_val, bsnn.accuracy, bsnn.label_cnt, bsnn.pred_cnt,
                 bsnn.pred_min, bsnn.pred_max,], feed_dict=feed_dict)

            #print('train_label_cnt: ', label_cnt)
            #print('train_min_max:', pred_min, pred_max)
            #print('train_cnt: ', pred_cnt)
            time_str = datetime.datetime.now().isoformat()
            print("{}:step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return step

        def dev_step(dev_x, dev_y):
            feed_dict = {bsnn.input_x: dev_x,
                         bsnn.input_y: dev_y,
                         bsnn.dropout_keep_prob: 1.0
                         }
            step, loss, accuracy, label_cnt, pred_cnt = sess.run(
                [bsnn.global_step, bsnn.loss_val, bsnn.accuracy, bsnn.label_cnt, bsnn.pred_cnt,], feed_dict=feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("dev result: {}:step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


        for epoch in range(FLAGS.num_epochs):
            print('current epoch %s' % (epoch + 1))
            for i in range(0, 270, FLAGS.batch_size):
                x = train_x[i:i + FLAGS.batch_size]
                y = train_y[i:i + FLAGS.batch_size]
                train_step(x, y)
                cur_step = tf.train.global_step(sess, bsnn.global_step)

            print('\n')
            dev_step(dev_x, dev_y)
            path = saver.save(sess, checkpoint_prefix, global_step=epoch)
            print('Saved model checkpoint to {}\n'.format(path))
