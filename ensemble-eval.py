import tensorflow as tf
import numpy as np
import os
import data_helpers

# use BioSent2Vec and traditional ml on titles
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score
import glob
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
import sent2vec

neg_path = '/home/paperspace/Documents/RCNN-421-BioSentVec/data/title.neg-ns'
pos_path = '/home/paperspace/Documents/RCNN-421-BioSentVec/data/title.pos-ns'
model_path = '/home/paperspace/Documents/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
model = sent2vec.Sent2vecModel()

try:
    model.load_model(model_path)
except Exception as e:
    print(e)
print('model successfully loaded')

# load classic ml models
abs_dir = "/home/paperspace/Documents/BioSentVec/"
lr_dir = abs_dir + "lg_models/"
rf_dir = abs_dir + "rf_models/"
svm_dir = abs_dir + "svm_models_linear/"# svm_models_linear/c_0.1_svm.sav

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("pos_dir", "data/rt-polaritydata/rt-polarity.pos", "Path of positive data")
tf.flags.DEFINE_string("neg_dir", "data/rt-polaritydata/rt-polarity.neg", "Path of negative data")
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of all data to use for validation") #TJ test
tf.flags.DEFINE_float("test_sample_percentage", .2, "Percentage of all data to use for test") #TJ test
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")

def eval():
    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(FLAGS.pos_dir, FLAGS.neg_dir)

    # Map data into vocabulary
    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    x_eval = np.array(list(text_vocab_processor.transform(x_text)))
    y_eval = np.argmax(y, axis=1)
    
    x = x_eval
    y = y_eval
    
    pos_arrays, neg_arrays = np.array_split(x, 2)
    pos_labels, neg_labels = np.array_split(y, 2)

    # prepare the data from one single file
    # 60/20/20
    pos_arrays_700 = np.zeros((23499, 700)) # the sum of all the training examples (neg+pos)
    pos_labels_700 = np.zeros(23499)

    neg_arrays_700 = np.zeros((23499, 700)) # the sum of all the training examples (neg+pos)
    neg_labels_700 = np.zeros(23499) 

    with open(FLAGS.neg_dir) as neg:
        i = 0
        for line in neg:
    	    sentence_vector = model.embed_sentence(line)
    	    neg_arrays_700[i] = sentence_vector
    	    neg_labels_700[i] = 0
    	    i = i + 1
    
    with open(FLAGS.pos_dir) as pos:
        i = 0
        for line in pos:
    	    sentence_vector = model.embed_sentence(line)
    	    pos_arrays_700[i] = sentence_vector
    	    pos_labels_700[i] = 1
    	    i = i + 1

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

    np.random.shuffle(pos_arrays_700)
    np.random.shuffle(neg_arrays_700)
    np.random.shuffle(pos_labels_700)
    np.random.shuffle(neg_labels_700)
    training_pos_700, val_pos_700, test_pos_700  = pos_arrays_700[:dev_sample_index+test_sample_index],  pos_arrays_700[dev_sample_index+test_sample_index:test_sample_index], pos_arrays_700[test_sample_index:] 
    training_neg_700, val_neg_700, test_neg_700  = neg_arrays_700[:dev_sample_index+test_sample_index], neg_arrays_700[dev_sample_index+test_sample_index:test_sample_index], neg_arrays_700[test_sample_index:] 

    training_pos_label_700, val_pos_label_700, test_pos_label_700  = pos_labels_700[:dev_sample_index+test_sample_index],  pos_labels_700[dev_sample_index+test_sample_index:test_sample_index], pos_labels_700[test_sample_index:] 
    training_neg_label_700, val_neg_label_700, test_neg_label_700  = neg_labels_700[:dev_sample_index+test_sample_index], neg_labels_700[dev_sample_index+test_sample_index:test_sample_index], neg_labels_700[test_sample_index:]

    x_train_700 = np.concatenate((training_pos_700, training_neg_700))
    x_dev_700 = np.concatenate((val_pos_700, val_neg_700))
    x_test_700 = np.concatenate((test_pos_700, test_neg_700))

    y_train_700 = np.concatenate((training_pos_label_700, training_neg_label_700))
    y_dev_700 = np.concatenate((val_pos_label_700, val_neg_label_700))
    y_test_700 = np.concatenate((test_pos_label_700, test_neg_label_700))


    print("Train/Dev/Test split: {:d}/{:d}/{:d}\n".format(len(y_train), len(y_dev),len(y_test)))  #TJ test
    
    x_eval = x_test
    y_eval = y_test

    lr_model_1 = pickle.load(open(lr_dir + "0.01_lr.sav", 'rb')) #0.01
    lr_model_2 = pickle.load(open(lr_dir + "0.1_lr.sav", 'rb'))
    svm_model_1 = pickle.load(open(svm_dir + "c_0.1_svm.sav", 'rb'))
    svm_model_2 = pickle.load(open(svm_dir + "c_0.01_svm.sav", 'rb'))

    Y_test_pred_lr_1 = (lr_model_1.predict_proba(x_test_700) >= 0.25).astype(int) #0.3,,
    Y_test_pred_lr_1 = Y_test_pred_lr_1[:, 1]
    Y_test_pred_s1 = svm_model_1.predict(x_test_700)

    Y_test_pred = np.add(Y_test_pred_lr_1, Y_test_pred_s1)

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_text: x_batch,
                                                           dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            correct_predictions = float(sum(all_predictions == y_eval))
            print("Total number of test examples: {}".format(len(y_eval)))
            Y_test_pred = np.add(Y_test_pred.astype(int), all_predictions.astype(int))
            #Y_test_pred[Y_test_pred >= 2.0] = 1.0
            Y_test_pred[Y_test_pred < 2] = 0
            Y_test_pred[Y_test_pred > 0] = 1

            Y_test_pred=Y_test_pred.astype(int) 
            y_eval=y_eval.astype(int) 
            acc_test, pre_test, rec_test, f1_test = accuracy_score(y_eval, Y_test_pred), precision_score(y_eval, Y_test_pred), recall_score(y_eval, Y_test_pred), f1_score(y_eval, Y_test_pred)
            print("accuracy = " + str(acc_test))
            print("precison = " + str(pre_test))
            print("recall = " + str(rec_test))
            print("f1 = " + str(f1_test))


def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()
