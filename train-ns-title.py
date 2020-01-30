import tensorflow as tf
import numpy as np
import os
import datetime
import time
from rcnn import TextRCNN
import data_helpers
#TJ for output logs
import logging 

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("pos_dir", "data/rt-polaritydata/rt-polarity.pos-ns", "Path of positive data")
tf.flags.DEFINE_string("neg_dir", "data/rt-polaritydata/rt-polarity.neg-ns", "Path of negative data")
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of all data to use for validation") #TJ test
tf.flags.DEFINE_float("test_sample_percentage", .2, "Percentage of all data to use for test") #TJ test
tf.flags.DEFINE_integer("max_sentence_length", 50, "Max sentence length in train/test data (Default: 50)")

# Model Hyperparameters
tf.flags.DEFINE_string("cell_type", "vanilla", "Type of RNN cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings")
tf.flags.DEFINE_integer("word_embedding_dim", 300, "Dimensionality of word embedding (Default: 300)")
tf.flags.DEFINE_integer("context_embedding_dim", 110, "Dimensionality of context embedding(= RNN state size)  (Default: 512)")
tf.flags.DEFINE_integer("hidden_size", 110, "Size of hidden layer (Default: 512)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability (Default: 0.7)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda (Default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (Default: 32)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (Default: 10)")
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev and test set after this many steps") #TJ test
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store") #TJ original 5
tf.flags.DEFINE_float("learning_rate", 1e-2, "Which learning rate to start with. (Default: 1e-2)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def train():
    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(FLAGS.pos_dir, FLAGS.neg_dir)

    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    x = np.array(list(text_vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))

    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")
    print(type(x))
    print(type(y))
    pos_arrays, neg_arrays = np.array_split(x, 2)
    pos_labels, neg_labels = np.array_split(y, 2)

    # Randomly shuffle data
    np.random.seed(10)
    total = len(y) / 2 
    np.random.shuffle(pos_arrays)
    np.random.shuffle(neg_arrays)
    np.random.shuffle(pos_labels)
    np.random.shuffle(neg_labels)
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(total))
    test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(total)) 
    training_pos, val_pos, test_pos  = pos_arrays[:dev_sample_index+test_sample_index],  pos_arrays[dev_sample_index+test_sample_index:test_sample_index], pos_arrays[test_sample_index:] 
    training_neg, val_neg, test_neg  = neg_arrays[:dev_sample_index+test_sample_index], neg_arrays[dev_sample_index+test_sample_index:test_sample_index], neg_arrays[test_sample_index:] 

    training_pos_label, val_pos_label, test_pos_label  = pos_labels[:dev_sample_index+test_sample_index],  pos_labels[dev_sample_index+test_sample_index:test_sample_index], pos_labels[test_sample_index:] 
    training_neg_label, val_neg_label, test_neg_label  = neg_labels[:dev_sample_index+test_sample_index], neg_labels[dev_sample_index+test_sample_index:test_sample_index], neg_labels[test_sample_index:]

    x_train = np.concatenate((training_pos, training_neg))
    x_dev = np.concatenate((val_pos, val_neg))
    x_test = np.concatenate((test_pos, test_neg))

    y_train = np.concatenate((training_pos_label, training_neg_label))
    y_dev = np.concatenate((val_pos_label, val_neg_label))
    y_test = np.concatenate((test_pos_label, test_neg_label))

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train_shuffled = x_train[shuffle_indices]
    y_train_shuffled = y_train[shuffle_indices]
    x_train = x_train_shuffled
    y_train = y_train_shuffled

    print("Train/Dev/Test split: {:d}/{:d}/{:d}\n".format(len(y_train), len(y_dev),len(y_test)))  #TJ test

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    '''
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))  #TJ test
    test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(y)))  #TJ test
    x_train, x_dev, x_test  = x_shuffled[:dev_sample_index+test_sample_index], x_shuffled[dev_sample_index+test_sample_index:test_sample_index], x_shuffled[test_sample_index:] #TJ test
    y_train, y_dev, y_test = y_shuffled[:dev_sample_index+test_sample_index], y_shuffled[dev_sample_index+test_sample_index:test_sample_index], y_shuffled[test_sample_index:] #TJ test
    '''   
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rcnn = TextRCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(text_vocab_processor.vocabulary_),
                word_embedding_size=FLAGS.word_embedding_dim,
                context_embedding_size=FLAGS.context_embedding_dim,
                cell_type=FLAGS.cell_type,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(rcnn.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs-ns", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rcnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rcnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Test summaries-TJ
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.word2vec:
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (len(text_vocab_processor.vocabulary_), FLAGS.word_embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec file {0}".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in range(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1).decode('latin-1')
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = text_vocab_processor.vocabulary_.get(word)
                        if idx != 0:
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)
                sess.run(rcnn.W_text.assign(initW))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    rcnn.input_text: x_batch,
                    rcnn.input_y: y_batch,
                    rcnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, rcnn.loss, rcnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict_dev = {
                        rcnn.input_text: x_dev,
                        rcnn.input_y: y_dev,
                        rcnn.dropout_keep_prob: 1.0
                    } #TJ precision, recall, f1
                    summaries_dev, loss, accuracy, precision, recall, f1= sess.run(
                        [dev_summary_op, rcnn.loss, rcnn.accuracy, rcnn.precision, rcnn.recall, rcnn.f1], feed_dict_dev)

                    dev_summary_writer.add_summary(summaries_dev, step)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}, f1 {:g}\n".format(time_str, step, loss, accuracy, precision, recall, f1))

               # Test - TJ
                if step % FLAGS.evaluate_every == 0:
                    print("\nTesting:")
                    feed_dict_test = {
                        rcnn.input_text: x_test,
                        rcnn.input_y: y_test,
                        rcnn.dropout_keep_prob: 1.0
                    } #TJ precision, recall, f1
                    summaries_test, loss, accuracy, precision, recall, f1= sess.run(
                        [test_summary_op, rcnn.loss, rcnn.accuracy, rcnn.precision, rcnn.recall, rcnn.f1], feed_dict_test)

                    test_summary_writer.add_summary(summaries_test, step)

                    time_str = datetime.datetime.now().isoformat()
                    print("test part {}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}, f1 {:g}\n".format(time_str, step, loss, accuracy, precision, recall, f1))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
